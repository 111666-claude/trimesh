"""
Regression coverage for preserving common MTL properties
(Ni / d / Tr / illum) and original material names through the
OBJ load -> export -> load roundtrip.

Two materials share identical colors but have different
refractive indices and must not be merged on export.
"""

import numpy as np

import trimesh

# two fixed MTL samples: same Kd/Ka/Ks/Ns, different Ni
MTL_TEXT = """\
newmtl GlassLow
Ka 0.1 0.1 0.1
Kd 0.8 0.2 0.2
Ks 0.5 0.5 0.5
Ns 96.078431
Ni 1.5
d 1.0
illum 2

newmtl GlassHigh
Ka 0.1 0.1 0.1
Kd 0.8 0.2 0.2
Ks 0.5 0.5 0.5
Ns 96.078431
Ni 2.4
Tr 0.25
illum 4
"""

OBJ_TEXT = """\
mtllib samples.mtl
o Low
usemtl GlassLow
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
o High
usemtl GlassHigh
v 2 0 0
v 3 0 0
v 3 1 0
f 4 5 6
"""


def _write_samples(path):
    with open(path / "samples.obj", "w") as f:
        f.write(OBJ_TEXT)
    with open(path / "samples.mtl", "w") as f:
        f.write(MTL_TEXT)


def _materials(scene):
    return {name: geom.visual.material for name, geom in scene.geometry.items()}


def test_two_same_color_different_ni_load(tmp_path):
    _write_samples(tmp_path)
    scene = trimesh.load(str(tmp_path / "samples.obj"))
    mats = _materials(scene)

    assert set(mats) == {"GlassLow", "GlassHigh"}
    low, high = mats["GlassLow"], mats["GlassHigh"]

    # original MTL names are preserved on the materials
    assert low.name == "GlassLow"
    assert high.name == "GlassHigh"

    # parsed numeric values survive on the material objects
    assert np.isclose(low.index_of_refraction, 1.5)
    assert np.isclose(high.index_of_refraction, 2.4)
    assert low.illumination_model == 2
    assert high.illumination_model == 4
    assert np.isclose(low.dissolve, 1.0)
    # Tr 0.25 (transparency) maps to d == 0.75 (opacity)
    assert np.isclose(high.dissolve, 0.75)

    # same colors but different Ni must hash differently
    assert np.allclose(low.diffuse, high.diffuse)
    assert hash(low) != hash(high)


def test_two_ni_samples_export_and_roundtrip(tmp_path):
    _write_samples(tmp_path)
    scene = trimesh.load(str(tmp_path / "samples.obj"))

    obj_text, mtl_files = trimesh.exchange.obj.export_obj(scene, return_texture=True)
    mtl_text = mtl_files["material.mtl"].decode("utf-8")

    # OBJ distinguishes both materials by their original usemtl names
    assert obj_text.count("usemtl GlassLow") == 1
    assert obj_text.count("usemtl GlassHigh") == 1

    # MTL keeps two named materials and their Ni / illum values
    assert mtl_text.count("newmtl GlassLow") == 1
    assert mtl_text.count("newmtl GlassHigh") == 1
    assert "Ni 1.50000000" in mtl_text
    assert "Ni 2.40000000" in mtl_text
    assert "illum 2" in mtl_text
    assert "illum 4" in mtl_text
    # Tr was converted to the equivalent d (opacity)
    assert "d 0.75000000" in mtl_text

    # write the export next to the original-style layout and read back
    out_obj = tmp_path / "out.obj"
    out_obj.write_text(obj_text)
    for file_name, data in mtl_files.items():
        (tmp_path / file_name).write_bytes(data)

    reloaded = trimesh.load(str(out_obj))
    mats = _materials(reloaded)
    assert set(mats) == {"GlassLow", "GlassHigh"}
    low, high = mats["GlassLow"], mats["GlassHigh"]
    assert np.isclose(low.index_of_refraction, 1.5)
    assert np.isclose(high.index_of_refraction, 2.4)
    assert low.illumination_model == 2
    assert high.illumination_model == 4
    assert np.isclose(high.dissolve, 0.75)
    assert hash(low) != hash(high)

    # a second export stays stable
    obj2, mtl2 = trimesh.exchange.obj.export_obj(reloaded, return_texture=True)
    assert mtl2["material.mtl"].decode().count("newmtl") == 2
    assert obj2.count("usemtl GlassLow") == 1
    assert obj2.count("usemtl GlassHigh") == 1


def test_material_without_extra_properties_exports_as_before():
    mat = trimesh.visual.material.SimpleMaterial(
        diffuse=[204, 51, 51, 255],
        ambient=[26, 26, 26, 255],
        specular=[128, 128, 128, 255],
        glossiness=96.078431,
        name="Plain",
    )
    data, name = mat.to_obj()
    lines = data[f"{name}.mtl"].decode().splitlines()

    # legacy Ka/Kd/Ks/Ns directives remain and no new lines appear
    assert lines[0] == "newmtl Plain"
    assert any(line.startswith("Ka ") for line in lines)
    assert any(line.startswith("Kd ") for line in lines)
    assert any(line.startswith("Ks ") for line in lines)
    assert any(line.startswith("Ns ") for line in lines)
    assert not any(
        line.startswith(("Ni ", "d ", "Tr ", "Tf ", "illum ")) for line in lines
    )
