import io
import uuid
import zipfile
from collections import defaultdict

import numpy as np

from .. import graph, util
from ..constants import log
from ..util import unique_name


def _read_mesh(mesh):
    """
    Read a `<mesh ` XML element into Numpy vertices and faces.

    This is generally the most expensive operation in the load as it
    has to operate in Python-space on every single vertex and face.

    Parameters
    ----------
    mesh : lxml.etree.Element
      Input mesh element with `vertex` and `triangle` children.

    Returns
    ----------
    vertex_array : (n, 3) float64
      Vertices
    face_array : (n, 3) int64
      Indexes of vertices forming triangles.
    face_pid : (n,) int64
      Per-triangle property group (`pid`) override: `0` means the
      triangle does not override the object default, positive values
      are resource ids and `-1` marks an unparsable explicit id.
    face_pindex : (n,) int64
      Per-triangle property index (`p1`, falling back to `p2`/`p3`),
      `-1` when unspecified.
    """
    # get the XML elements for vertices and faces
    vertices = mesh.find("{*}vertices")
    faces = mesh.find("{*}triangles")

    # get every value as a flat space-delimited string
    # this is very sensitive as it is large, i.e. it is
    # much faster with the full list comprehension before
    # the `.join` as the giant string can be fully allocated
    vs = " ".join(
        [
            f"{i.attrib['x']} {i.attrib['y']} {i.attrib['z']}"
            for i in vertices.iter("{*}vertex")
        ]
    )
    # convert every value to floating point in one-shot rather than in a loop
    v_array = np.fromstring(vs, dtype=np.float64, sep=" ").reshape((-1, 3))

    # do the same behavior for faces but as an integer, collecting the
    # optional `pid`/`p1` material overrides in the same pass
    face_tokens = []
    face_pid = []
    face_pindex = []
    for tri in faces.iter("{*}triangle"):
        attrib = tri.attrib
        face_tokens.append(f"{attrib['v1']} {attrib['v2']} {attrib['v3']}")

        pid = attrib.get("pid")
        if pid is None:
            face_pid.append(0)
        else:
            try:
                face_pid.append(int(pid))
            except (TypeError, ValueError):
                # explicit but not a valid resource id: never resolve
                face_pid.append(-1)

        # for base materials p1 == p2 == p3 so p1 is the face index
        index = attrib.get("p1", attrib.get("p2", attrib.get("p3")))
        try:
            face_pindex.append(int(index) if index is not None else -1)
        except (TypeError, ValueError):
            face_pindex.append(-1)

    f_array = np.fromstring(" ".join(face_tokens), dtype=np.int64, sep=" ").reshape(
        (-1, 3)
    )

    return (
        v_array,
        f_array,
        np.asarray(face_pid, dtype=np.int64),
        np.asarray(face_pindex, dtype=np.int64),
    )


def _parse_display_color(value):
    """
    Parse a 3MF `displaycolor` (`#RRGGBB` or `#RRGGBBAA`) value.

    Parameters
    ----------
    value : str or None
      Hex encoded sRGB color, alpha is the final byte and defaults to 255.

    Returns
    ----------
    rgba : (4,) uint8 or None
      Parsed color, or `None` if the value is missing/malformed.
    """
    if value is None:
        return None
    text = str(value).strip().lstrip("#")
    try:
        if len(text) == 6:
            rgba = [int(text[i : i + 2], 16) for i in (0, 2, 4)] + [255]
        elif len(text) == 8:
            rgba = [int(text[i : i + 2], 16) for i in (0, 2, 4, 6)]
        else:
            return None
    except ValueError:
        return None
    return np.asarray(rgba, dtype=np.uint8)


def _format_display_color(color):
    """
    Encode an RGBA color as a 3MF `displaycolor`, using the compact
    `#RRGGBB` form for opaque colors and `#RRGGBBAA` otherwise.
    """
    r, g, b, a = (int(c) for c in color)
    if a >= 255:
        return f"#{r:02X}{g:02X}{b:02X}"
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def _face_color_palette(mesh):
    """
    Collect the unique face colors of a mesh for base-material export.

    Returns
    ----------
    palette : (k, 4) uint8 or None
      Unique colors, or `None` if the mesh has no explicit face colors.
    inverse : (len(mesh.faces),) int64 or None
      Index into `palette` for every face.
    """
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "face":
        return None, None

    face_colors = np.asarray(visual.face_colors, dtype=np.uint8)
    if face_colors.ndim != 2 or face_colors.shape[1] != 4:
        return None, None
    if len(face_colors) != len(mesh.faces) or len(face_colors) == 0:
        return None, None

    palette, first, sorted_inverse = np.unique(
        face_colors, axis=0, return_index=True, return_inverse=True
    )
    # order the palette by first face occurrence rather than byte order
    order = np.argsort(first)
    palette = palette[order].astype(np.uint8)
    remap = np.zeros(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    inverse = remap[sorted_inverse].astype(np.int64)
    return palette, inverse


def _resolve_face_colors(
    count,
    face_pid,
    face_pindex,
    default_pid,
    default_pindex,
    material_groups,
    label="",
):
    """
    Resolve 3MF base-material references to one RGBA color per face.

    Parameters
    ----------
    count : int
      Number of faces.
    face_pid : (count,) int64
      Per-triangle pid override from `_read_mesh`.
    face_pindex : (count,) int64
      Per-triangle property index override.
    default_pid : int or None
      Object-level default material resource id.
    default_pindex : int
      Object-level default property index (`-1` if unspecified).
    material_groups : dict
      `{resource id int : (k, 4) uint8 palette}` for `<basematerials>`.
    label : str
      Object id/name used in warning messages.

    Returns
    ----------
    colors : (count, 4) uint8 or None
      One color per face (unresolved faces use the trimesh default),
      or `None` when no face referenced a resolvable base material.
    """
    from ..visual.color import DEFAULT_COLOR

    colors = np.tile(DEFAULT_COLOR, (count, 1)).astype(np.uint8)
    if count == 0:
        return None

    # effective resource group and index per face: triangle attributes
    # override the object-level pid/pindex defaults
    object_pid = int(default_pid) if default_pid is not None else 0
    object_index = int(default_pindex) if default_pindex is not None else -1

    effective_pid = np.where(face_pid != 0, face_pid, object_pid)
    effective_index = np.where(face_pindex >= 0, face_pindex, object_index)

    resolved_any = False
    for gid in np.unique(effective_pid):
        gid = int(gid)
        if gid <= 0:
            # gid == 0: no material specified; gid < 0: malformed pid
            if gid < 0:
                log.warning(
                    "3MF object %s has a triangle with an invalid pid; leaving that face with the default color",
                    label,
                )
            continue

        mask = effective_pid == gid
        palette = material_groups.get(gid)
        if palette is None:
            log.warning(
                "3MF object %s references material resource id %s which is not defined as a <basematerials> group; leaving %d face(s) with the default color",
                label,
                gid,
                int(mask.sum()),
            )
            continue

        indices = effective_index[mask]
        valid = (indices >= 0) & (indices < len(palette))
        if not valid.all():
            log.warning(
                "3MF object %s uses an out-of-range material index into resource %s (group has %d entries); leaving %d face(s) with the default color",
                label,
                gid,
                len(palette),
                int((~valid).sum()),
            )

        positions = np.flatnonzero(mask)
        if valid.any():
            colors[positions[valid]] = palette[indices[valid]]
            resolved_any = True

    return colors if resolved_any else None


def load_3MF(file_obj, postprocess=True, **kwargs):
    """
    Load a 3MF formatted file into a Trimesh scene.

    Parameters
    ------------
    file_obj : file-like
      Contains 3MF formatted data

    Returns
    ------------
    kwargs : dict
      Constructor arguments for `trimesh.Scene`
    """

    # dict, {name in archive: BytesIo}
    archive = util.decompress(file_obj, file_type="zip")
    # get model with case-insensitive keys
    model = next(iter(v for k, v in archive.items() if "3d/3dmodel.model" in k.lower()))

    # read root attributes only from XML first
    _event, root = next(
        etree.iterparse(model, tag=("{*}model"), events=("start",), **XML_PARSER_OPTIONS)
    )
    # collect unit information from the tree
    if "unit" in root.attrib:
        metadata = {"units": root.attrib["unit"]}
    else:
        # the default units, defined by the specification
        metadata = {"units": "millimeters"}

    # { mesh id : mesh name}
    id_name = {}
    # { mesh id: (n,3) float vertices}
    v_seq = defaultdict(list)
    # { mesh id: (n,3) int faces}
    f_seq = defaultdict(list)
    # components are objects that contain other objects
    # {id : [other ids]}
    components = defaultdict(list)
    # load information about the scene graph
    # each instance is a single geometry
    build_items = []

    # base material resources: {resource id : (k, 4) uint8 palette}
    material_groups = {}
    # object-level default material: {object id : (pid int|None, pindex int)}
    object_materials = {}
    # per object, per mesh part triangle overrides
    # {mesh id : [(pid array, pindex array), ...]}
    material_refs = defaultdict(list)

    # keep track of names we can use
    consumed_counts = {}
    consumed_names = set()

    # iterate the XML object and build elements with an LXML iterator
    # loaded elements are cleared to avoid ballooning memory
    model.seek(0)
    for _, obj in etree.iterparse(
        model,
        tag=("{*}object", "{*}build", "{*}basematerials"),
        events=("end",),
        **XML_PARSER_OPTIONS,
    ):
        # parse base material resource groups
        if "basematerials" in obj.tag:
            gid = obj.attrib.get("id")
            try:
                gid = int(gid)
            except (TypeError, ValueError):
                log.warning("3MF <basematerials> missing valid id: %s", gid)
                continue

            palette = []
            for base in obj.iter("{*}base"):
                rgba = _parse_display_color(base.attrib.get("displaycolor"))
                if rgba is None:
                    log.warning(
                        "3MF basematerials %s contains an invalid displaycolor %r; using the default color",
                        gid,
                        base.attrib.get("displaycolor"),
                    )
                    from ..visual.color import DEFAULT_COLOR

                    rgba = DEFAULT_COLOR
                palette.append(rgba)
            material_groups[gid] = (
                np.asarray(palette, dtype=np.uint8).reshape((-1, 4))
                if len(palette) > 0
                else np.zeros((0, 4), dtype=np.uint8)
            )

        # parse objects
        if "object" in obj.tag:
            # id is mandatory
            index = obj.attrib["id"]

            # object-level default base material (pid + pindex)
            default_pid = obj.attrib.get("pid")
            try:
                default_pid = int(default_pid) if default_pid is not None else None
            except (TypeError, ValueError):
                default_pid = None
            try:
                default_pindex = int(obj.attrib.get("pindex"))
            except (TypeError, ValueError):
                default_pindex = -1
            object_materials[index] = (default_pid, default_pindex)

            # start with stored name
            # apparently some exporters name multiple meshes
            # the same thing so check to see if it's been used
            name = unique_name(
                obj.attrib.get("name", str(index)), consumed_names, consumed_counts
            )
            consumed_names.add(name)
            # store name reference on the index
            id_name[index] = name

            # if the object has actual geometry data parse here
            for mesh in obj.iter("{*}mesh"):
                v, f, f_pid, f_pindex = _read_mesh(mesh)
                v_seq[index].append(v)
                f_seq[index].append(f)
                material_refs[index].append((f_pid, f_pindex))

            # components are references to other geometries
            for c in obj.iter("{*}component"):
                mesh_index = c.attrib["objectid"]
                transform = _attrib_to_transform(c.attrib)
                components[index].append((mesh_index, transform))

                # if this references another file as the `path` attrib
                path = next(
                    (v.strip("/") for k, v in c.attrib.items() if k.endswith("path")),
                    None,
                )
                if path is not None and path in archive:
                    archive[path].seek(0)
                    name = unique_name(
                        obj.attrib.get("name", str(mesh_index)),
                        consumed_names,
                        consumed_counts,
                    )
                    consumed_names.add(name)
                    # store name reference on the index
                    id_name[mesh_index] = name

                    for _, m in etree.iterparse(
                        archive[path],
                        tag=("{*}mesh"),
                        events=("end",),
                        **XML_PARSER_OPTIONS,
                    ):
                        v, f, f_pid, f_pindex = _read_mesh(m)
                        v_seq[mesh_index].append(v)
                        f_seq[mesh_index].append(f)
                        material_refs[mesh_index].append((f_pid, f_pindex))

        # parse build
        if "build" in obj.tag:
            # scene graph information stored here, aka "build" the scene
            for item in obj.iter("{*}item"):
                # get a transform from the item's attributes
                transform = _attrib_to_transform(item.attrib)
                partnumber = item.attrib.get("partnumber", None)
                # the index of the geometry this item instantiates
                build_items.append((item.attrib["objectid"], transform, partnumber))

    # have one mesh per 3MF object
    # one mesh per geometry ID, store as kwargs for the object
    meshes = {}
    for gid in v_seq.keys():
        v, f = util.append_faces(v_seq[gid], f_seq[gid])
        name = id_name[gid]
        mesh_kwargs = {
            "vertices": v,
            "faces": f,
            "metadata": metadata.copy(),
        }

        # resolve core-spec base materials into one color per face
        refs = material_refs.get(gid)
        if refs:
            ref_pid = np.concatenate([pid for pid, _ in refs])
            ref_pindex = np.concatenate([pindex for _, pindex in refs])
            if len(ref_pid) != len(f):
                # should not happen, but never let malformed input
                # misalign colors with faces
                log.warning("3MF material references did not align with faces")
                ref_pid = np.zeros(len(f), dtype=np.int64)
                ref_pindex = np.full(len(f), -1, dtype=np.int64)

            default_pid, default_pindex = object_materials.get(gid, (None, -1))
            face_colors = _resolve_face_colors(
                count=len(f),
                face_pid=ref_pid,
                face_pindex=ref_pindex,
                default_pid=default_pid,
                default_pindex=default_pindex,
                material_groups=material_groups,
                label=str(gid),
            )
            if face_colors is not None:
                mesh_kwargs["face_colors"] = face_colors

        meshes[name] = mesh_kwargs
        # apply any keyword arguments that aren't None
        meshes[name].update({k: v for k, v in kwargs.items() if v is not None})

    # turn the item / component representation into
    # a MultiDiGraph to compound our pain
    g = nx.MultiDiGraph()
    # build items are the only things that exist according to 3MF
    # so we accomplish that by linking them to the base frame
    # if partnumbers are None, the key will be an int
    for gid, tf, partnumber in build_items:
        g.add_edge("world", gid, key=partnumber, matrix=tf)
    # components are instances which need to be linked to base
    # frame by a build_item
    for start, group in components.items():
        for gid, tf in group:
            g.add_edge(start, gid, matrix=tf)

    # turn the graph into kwargs for a scene graph
    # flatten the scene structure and simplify to
    # a single unique node per instance
    graph_args = []
    parents = defaultdict(set)
    used_names = set()
    for path in graph.multigraph_paths(G=g, source="world"):
        # collect all the transform on the path
        transforms = graph.multigraph_collect(G=g, traversal=path, attrib="matrix")
        # combine them into a single transform
        if len(transforms) == 1:
            transform = transforms[0]
        else:
            transform = util.multi_dot(transforms)

        # the last element of the path should be the geometry
        last = path[-1][0]
        # if someone included an undefined component, skip it
        if last not in id_name:
            log.warning(f"id {last} included but not defined!")
            continue

        if len(path[-1]) > 1 and isinstance(path[-1][1], str):
            # use the `partnumber` as the name
            name = path[-1][1]
        else:
            # use the name from the id
            name = id_name[last]

        # make the name unique
        name = unique_name(name, used_names)
        used_names.add(name)

        # index in meshes
        geom = id_name[last]

        # collect parents if we want to combine later
        if len(path) > 2:
            parent = path[-2][0]
            parents[parent].add(last)

        graph_args.append(
            {
                "frame_from": "world",
                "frame_to": name,
                "matrix": transform,
                "geometry": geom,
            }
        )

    # solidworks will export each body as its own mesh with the part
    # name as the parent so optionally rename and combine these bodies
    if postprocess and all("body" in i.lower() for i in meshes.keys()):
        # don't rename by default
        rename = {k: k for k in meshes.keys()}
        for parent, mesh_name in parents.items():
            # only handle the case where a parent has a single child
            # if there are multiple children we would do a combine op
            if len(mesh_name) != 1:
                continue
            # rename the part
            rename[id_name[next(iter(mesh_name))]] = id_name[parent].split("(")[0]

        # apply the rename operation meshes
        meshes = {rename[k]: m for k, m in meshes.items()}
        # rename geometry references in the scene graph
        for arg in graph_args:
            if "geometry" in arg:
                arg["geometry"] = rename[arg["geometry"]]

    # construct the kwargs to load the scene
    kwargs = {
        "base_frame": "world",
        "graph": graph_args,
        "geometry": meshes,
        "metadata": metadata,
    }

    return kwargs


def export_3MF(mesh, batch_size=4096, compression=zipfile.ZIP_DEFLATED, compresslevel=5):
    """
    Converts a Trimesh object into a 3MF file.

    Parameters
    ---------
    mesh trimesh.trimesh
      Mesh or Scene to export.
    batch_size : int
      Number of nodes to write per batch.
    compression : zipfile.ZIP_*
      Type of zip compression to use in this export.
    compresslevel : int
      Specify the 0-9 compression level.

    Returns
    ---------
    export : bytes
      Represents geometry as a 3MF file.
    """

    from ..scene.scene import Scene

    if not isinstance(mesh, Scene):
        mesh = Scene(mesh)

    geometry = mesh.geometry
    graph = mesh.graph.to_networkx()
    base_frame = mesh.graph.base_frame

    # xml namespaces
    model_nsmap = {
        None: "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
        "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
        "p": "http://schemas.microsoft.com/3dmanufacturing/production/2015/06",
        "b": "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02",
        "s": "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        "sc": "http://schemas.microsoft.com/3dmanufacturing/securecontent/2019/04",
    }

    rels_nsmap = {None: "http://schemas.openxmlformats.org/package/2006/relationships"}

    # model ids
    models = []

    def model_id(x):
        if x not in models:
            models.append(x)
        return str(models.index(x) + 1)

    def object_id(node, data):
        # nodes with children are written as a `components` object
        # which also references the node's own geometry
        if len(graph[node]) > 0:
            return model_id(("node", node))
        return model_id(data.get("geometry", node))

    # 3mf archive dict {path: BytesIO}
    file_obj = io.BytesIO()

    # specify the parameters for the zip container
    zip_kwargs = {"compression": compression, "compresslevel": compresslevel}

    with zipfile.ZipFile(file_obj, mode="w", **zip_kwargs) as z:
        # 3dmodel.model
        with (
            z.open("3D/3dmodel.model", mode="w") as f,
            etree.xmlfile(f, encoding="utf-8") as xf,
        ):
            xf.write_declaration()

            # stream elements
            with xf.element("model", {"unit": "millimeter"}, nsmap=model_nsmap):
                # objects with mesh data and/or references to other objects
                with xf.element("resources"):
                    # resolve face-color palettes up front: CT_Resources
                    # requires every property group to be written before
                    # the <object> elements
                    material_info = {}
                    for name, m in geometry.items():
                        palette, inverse = _face_color_palette(m)
                        if palette is not None:
                            material_info[name] = (
                                palette,
                                inverse,
                                model_id(("basematerials", name)),
                            )

                    # write one core-spec <basematerials> group per colored mesh
                    for palette, _inverse, material_id in material_info.values():
                        with xf.element("basematerials", {"id": material_id}):
                            xf.flush()
                            for start in range(0, len(palette), batch_size):
                                chunk = palette[start : start + batch_size]
                                bases = "".join(
                                    f'<base name="color_{start + j}" displaycolor="{_format_display_color(color)}" />'
                                    for j, color in enumerate(chunk)
                                )
                                f.write(bases.encode("utf-8"))

                    # stream objects with actual mesh data
                    for name, m in geometry.items():
                        info = material_info.get(name)
                        if info is not None:
                            palette, inverse, material_id = info
                        else:
                            palette = inverse = material_id = None

                        # attributes for object
                        attribs = {
                            "id": model_id(name),
                            "name": name,
                            "type": "model",
                            "p:UUID": str(uuid.uuid4()),
                        }
                        if material_id is not None:
                            # object-wide default material: uniform color
                            # for single-palette meshes, index zero otherwise
                            attribs["pid"] = material_id
                            attribs["pindex"] = "0"

                        with xf.element("object", **attribs):
                            with xf.element("mesh"):
                                with xf.element("vertices"):
                                    # vertex nodes are written directly to the file
                                    # so make sure lxml's buffer is flushed
                                    xf.flush()
                                    for start in range(0, len(m.vertices), batch_size):
                                        batch = m.vertices[start : start + batch_size]
                                        fragment = (
                                            '<vertex x="{}" y="{}" z="{}" />' * len(batch)
                                        )
                                        f.write(
                                            fragment.format(*batch.flatten()).encode(
                                                "utf-8"
                                            )
                                        )
                                with xf.element("triangles"):
                                    xf.flush()
                                    # more than one unique color means each
                                    # triangle references its palette entry
                                    per_face = (
                                        material_id is not None and len(palette) > 1
                                    )
                                    for start in range(0, len(m.faces), batch_size):
                                        batch = m.faces[start : start + batch_size]
                                        if per_face:
                                            indices = inverse[start : start + batch_size]
                                            fragment = (
                                                f'<triangle v1="{{}}" v2="{{}}" v3="{{}}" pid="{material_id}" p1="{{}}" />'
                                                * len(batch)
                                            )
                                            values = np.column_stack(
                                                (batch, indices)
                                            ).flatten()
                                        else:
                                            fragment = (
                                                '<triangle v1="{}" v2="{}" v3="{}" />'
                                                * len(batch)
                                            )
                                            values = batch.flatten()
                                        f.write(fragment.format(*values).encode("utf-8"))

                    # stream components
                    for node in graph.nodes:
                        if node == base_frame or node.startswith("camera"):
                            continue
                        if len(graph[node]) == 0:
                            continue

                        attribs = {
                            "id": model_id(("node", node)),
                            "name": node,
                            "type": "model",
                            "p:UUID": str(uuid.uuid4()),
                        }
                        with xf.element("object", **attribs):
                            with xf.element("components"):
                                geom = mesh.graph.transforms.node_data[node].get(
                                    "geometry"
                                )
                                if geom is not None:
                                    xf.write(
                                        etree.Element(
                                            "component", {"objectid": model_id(geom)}
                                        )
                                    )
                                for next, data in graph[node].items():
                                    transform = " ".join(
                                        str(i)
                                        for i in np.array(data["matrix"])[
                                            :3, :4
                                        ].T.flatten()
                                    )
                                    xf.write(
                                        etree.Element(
                                            "component",
                                            {
                                                "objectid": object_id(next, data),
                                                "transform": transform,
                                            },
                                        )
                                    )

                # stream build (objects on base_frame)
                with xf.element("build", {"p:UUID": str(uuid.uuid4())}):
                    for node, data in graph[base_frame].items():
                        if node.startswith("camera"):
                            continue
                        transform = " ".join(
                            str(i) for i in np.array(data["matrix"])[:3, :4].T.flatten()
                        )
                        uuid_tag = "{{{}}}UUID".format(model_nsmap["p"])
                        xf.write(
                            etree.Element(
                                "item",
                                {
                                    "objectid": object_id(node, data),
                                    "transform": transform,
                                    uuid_tag: str(uuid.uuid4()),
                                    "partnumber": node,
                                },
                                nsmap=model_nsmap,
                            )
                        )

        # .rels
        with z.open("_rels/.rels", "w") as f, etree.xmlfile(f, encoding="utf-8") as xf:
            xf.write_declaration()
            # stream elements
            with xf.element("Relationships", nsmap=rels_nsmap):
                rt = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
                xf.write(
                    etree.Element(
                        "Relationship",
                        Type=rt,
                        Target="/3D/3dmodel.model",
                        Id="rel0",
                    )
                )

        # [Content_Types].xml
        with (
            z.open("[Content_Types].xml", "w") as f,
            etree.xmlfile(f, encoding="utf-8") as xf,
        ):
            xf.write_declaration()
            # xml namespaces
            nsmap = {None: "http://schemas.openxmlformats.org/package/2006/content-types"}

            # stream elements
            types = [
                ("jpeg", "image/jpeg"),
                ("jpg", "image/jpeg"),
                ("model", "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"),
                ("png", "image/png"),
                ("rels", "application/vnd.openxmlformats-package.relationships+xml"),
                (
                    "texture",
                    "application/vnd.ms-package.3dmanufacturing-3dmodeltexture",
                ),
            ]
            with xf.element("Types", nsmap=nsmap):
                for ext, ctype in types:
                    xf.write(etree.Element("Default", Extension=ext, ContentType=ctype))

    return file_obj.getvalue()


def _attrib_to_transform(attrib):
    """
    Extract a homogeneous transform from a dictionary.

    Parameters
    ------------
    attrib: dict, optionally containing 'transform'

    Returns
    ------------
    transform: (4, 4) float, homogeonous transformation
    """

    transform = np.eye(4, dtype=np.float64)
    if "transform" in attrib:
        # wangle their transform format
        values = np.array(attrib["transform"].split(), dtype=np.float64).reshape((4, 3)).T
        transform[:3, :4] = values
    return transform


# do import here to keep lxml a soft dependency
try:
    import networkx as nx
    from lxml import etree

    from .common import XML_PARSER_OPTIONS

    _three_loaders = {"3mf": load_3MF}
    _3mf_exporters = {"3mf": export_3MF}
except BaseException as E:
    from ..exceptions import ExceptionWrapper

    _three_loaders = {"3mf": ExceptionWrapper(E)}
    _3mf_exporters = {"3mf": ExceptionWrapper(E)}
