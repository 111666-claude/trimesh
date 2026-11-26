"""
voxel.py
-----------

Convert meshes to a simple voxel data structure and back again.
"""

import numpy as np

from .. import bounds as bounds_module
from .. import caching, util
from .. import transformations as tr
from ..constants import log
from ..exchange.binvox import export_binvox
from ..parent import Geometry
from . import morphology, ops, transforms
from .encoding import DenseEncoding, Encoding, SparseBinaryEncoding


class VoxelGrid(Geometry):
    """
    Store 3D voxels.
    """

    def __init__(self, encoding, transform=None, metadata=None):
        if transform is None:
            transform = np.eye(4)
        if isinstance(encoding, np.ndarray):
            encoding = DenseEncoding(encoding.astype(bool))
        if encoding.dtype != bool:
            raise ValueError("encoding must have dtype bool")
        self._data = caching.DataStore()
        self.encoding = encoding
        self._transform = transforms.Transform(transform, datastore=self._data)
        self._cache = caching.Cache(id_function=self._data.__hash__)

        self.metadata = {}
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        elif metadata is not None:
            raise ValueError(f"metadata should be a dict or None, got {metadata!s}")

    def __hash__(self):
        """
        Get the hash of the current transformation matrix.

        Returns
        ------------
        hash : str
          Hash of transformation matrix
        """
        return self._data.__hash__()

    @property
    def encoding(self):
        """
        `Encoding` object providing the occupancy grid.

        See `trimesh.voxel.encoding` for implementations.
        """
        return self._data["encoding"]

    @encoding.setter
    def encoding(self, encoding):
        if isinstance(encoding, np.ndarray):
            encoding = DenseEncoding(encoding)
        elif not isinstance(encoding, Encoding):
            raise ValueError(f"encoding must be an Encoding, got {encoding!s}")
        if len(encoding.shape) != 3:
            raise ValueError(f"encoding must be rank 3, got shape {encoding.shape!s}")
        if encoding.dtype != bool:
            raise ValueError(f"encoding must be binary, got {encoding.dtype}")
        self._data["encoding"] = encoding

    @property
    def transform(self):
        """4x4 homogeneous transformation matrix."""
        return self._transform.matrix

    @transform.setter
    def transform(self, matrix):
        """4x4 homogeneous transformation matrix."""
        self._transform.matrix = matrix

    @property
    def translation(self):
        """Location of voxel at [0, 0, 0]."""
        return self._transform.translation

    @property
    def scale(self):
        """
        3-element float representing per-axis scale.

        Raises a `RuntimeError` if `self.transform` has rotation or
        shear components.
        """
        return self._transform.scale

    @property
    def pitch(self):
        """
        Uniform scaling factor representing the side length of
        each voxel.

        Returns
        -----------
        pitch : float
          Pitch of the voxels.

        Raises
        ------------
        `RuntimeError`
          If `self.transformation` has rotation or shear
          components of has non-uniform scaling.
        """
        return self._transform.pitch

    @property
    def element_volume(self):
        return self._transform.unit_volume

    def apply_transform(self, matrix):
        self._transform.apply_transform(matrix)
        return self

    def strip(self):
        """
        Mutate self by stripping leading/trailing planes of zeros.

        Returns
        --------
        self after mutation occurs in-place
        """
        encoding, padding = self.encoding.stripped
        self.encoding = encoding
        self._transform.matrix[:3, 3] = self.indices_to_points(padding[:, 0])
        return self

    @caching.cache_decorator
    def bounds(self):
        indices = self.sparse_indices
        # get all 8 corners of the AABB
        corners = bounds_module.corners(
            [indices.min(axis=0) - 0.5, indices.max(axis=0) + 0.5]
        )
        # transform these corners to a new frame
        corners = self._transform.transform_points(corners)
        # get the AABB of corners in-frame
        bounds = np.array([corners.min(axis=0), corners.max(axis=0)])
        bounds.flags.writeable = False
        return bounds

    @caching.cache_decorator
    def extents(self):
        bounds = self.bounds
        extents = bounds[1] - bounds[0]
        extents.flags.writeable = False
        return extents

    @caching.cache_decorator
    def is_empty(self):
        return self.encoding.is_empty

    @property
    def shape(self):
        """3-tuple of ints denoting shape of occupancy grid."""
        return self.encoding.shape

    @caching.cache_decorator
    def filled_count(self):
        """int, number of occupied voxels in the grid."""
        return self.encoding.sum.item()

    def is_filled(self, point):
        """
        Query points to see if the voxel cells they lie in are
        filled or not.

        Parameters
        ----------
        point : (n, 3) float
          Points in space

        Returns
        ---------
        is_filled : (n,) bool
          Is cell occupied or not for each point
        """
        point = np.asanyarray(point)
        indices = self.points_to_indices(point)
        in_range = np.logical_and(
            np.all(indices < np.array(self.shape), axis=-1), np.all(indices >= 0, axis=-1)
        )

        is_filled = np.zeros_like(in_range)
        is_filled[in_range] = self.encoding.gather_nd(indices[in_range])
        return is_filled

    def fill(self, method="holes", **kwargs):
        """
        Mutates self by filling in the encoding according
        to `morphology.fill`.

        Parameters
        ----------
        method : hashable
          Implementation key, one of
          `trimesh.voxel.morphology.fill.fillers` keys
        **kwargs : dict
          Additional kwargs passed through to
          the keyed implementation.

        Returns
        ----------
        self : VoxelGrid
          After replacing encoding with a filled version.
        """
        self.encoding = morphology.fill(self.encoding, method=method, **kwargs)
        return self

    def hollow(self):
        """
        Mutates self by removing internal voxels
        leaving only surface elements.

        Surviving elements are those in encoding that are
        adjacent to an empty voxel where adjacency is
        controlled by `structure`.

        Returns
        ----------
        self : VoxelGrid
          After replacing encoding with a surface version.
        """
        self.encoding = morphology.surface(self.encoding)
        return self

    @caching.cache_decorator
    def marching_cubes(self):
        """
        A marching cubes Trimesh representation of the voxels.

        No effort was made to clean or smooth the result in any way;
        it is merely the result of applying the scikit-image
        measure.marching_cubes function to self.encoding.dense.

        Returns
        ---------
        meshed : trimesh.Trimesh
          Representing the current voxel
          object as returned by marching cubes algorithm.
        """
        return ops.matrix_to_marching_cubes(matrix=self.matrix)

    @property
    def matrix(self):
        """
        Return a DENSE matrix of the current voxel encoding.

        Returns
        -------------
        dense : (a, b, c) bool
          Numpy array of dense matrix
          Shortcut to voxel.encoding.dense
        """
        return self.encoding.dense

    @caching.cache_decorator
    def volume(self):
        """
        What is the volume of the filled cells in the current
        voxel object.

        Returns
        ---------
        volume : float
          Volume of filled cells.
        """
        return self.filled_count * self.element_volume

    @caching.cache_decorator
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points : (self.filled, 3) float
          Points in space.
        """
        return self._transform.transform_points(self.sparse_indices.astype(float))

    @property
    def sparse_indices(self):
        """(n, 3) int array of sparse indices of occupied voxels."""
        return self.encoding.sparse_indices

    def as_boxes(self, colors=None, **kwargs):
        """
        A rough Trimesh representation of the voxels with a box
        for each filled voxel.

        Parameters
        ----------
        colors : None, (3,) or (4,) float or uint8
          (X, Y, Z, 3) or (X, Y, Z, 4) float or uint8
          Where matrix.shape == (X, Y, Z)

        Returns
        ---------
        mesh : trimesh.Trimesh
          Mesh with one box per filled cell.
        """

        if colors is not None:
            colors = np.asanyarray(colors)
            if colors.ndim == 4:
                encoding = self.encoding
                if colors.shape[:3] == encoding.shape:
                    # TODO jackd: more efficient implementation?
                    # encoding.as_mask?
                    colors = colors[encoding.dense]
                else:
                    log.warning("colors incorrect shape!")
                    colors = None
            elif colors.shape not in ((3,), (4,)):
                log.warning("colors incorrect shape!")
                colors = None

        mesh = ops.multibox(centers=self.sparse_indices.astype(float), colors=colors)

        mesh = mesh.apply_transform(self.transform)
        return mesh

    def points_to_indices(self, points):
        """
        Convert points to indices in the matrix array.

        Parameters
        ----------
        points: (n, 3) float, point in space

        Returns
        ---------
        indices: (n, 3) int array of indices into self.encoding
        """
        points = self._transform.inverse_transform_points(points)
        return np.round(points).astype(int)

    def indices_to_points(self, indices):
        return self._transform.transform_points(indices.astype(float))

    def show(self, *args, **kwargs):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        return self.as_boxes(kwargs.pop("colors", None)).show(*args, **kwargs)

    def copy(self):
        return VoxelGrid(self.encoding.copy(), self._transform.matrix.copy())

    def export(self, file_obj=None, file_type=None, **kwargs):
        """
        Export the current VoxelGrid.

        Parameters
        ------------
        file_obj : file-like or str
          File or file-name to export to.
        file_type : None or str
          Only 'binvox' currently supported.

        Returns
        ---------
        export : bytes
          Value of export.
        """
        if isinstance(file_obj, str) and file_type is None:
            file_type = util.split_extension(file_obj).lower()

        if file_type != "binvox":
            raise ValueError("only binvox exports supported!")

        exported = export_binvox(self, **kwargs)
        if hasattr(file_obj, "write"):
            file_obj.write(exported)
        elif isinstance(file_obj, str):
            with open(file_obj, "wb") as f:
                f.write(exported)
        return exported

    def revoxelized(self, shape):
        """
        Create a new VoxelGrid without rotations, reflections
        or shearing.

        Parameters
        ----------
        shape : (3, int)
          The shape of the returned VoxelGrid.

        Returns
        ----------
        vox : VoxelGrid
          Of the given shape with possibly non-uniform
          scale and translation transformation matrix.
        """
        shape = tuple(shape)
        bounds = self.bounds.copy()
        extents = self.extents
        points = util.grid_linspace(bounds, shape).reshape(shape + (3,))
        dense = self.is_filled(points)
        scale = extents / np.asanyarray(shape)
        translate = bounds[0]
        return VoxelGrid(dense, transform=tr.scale_and_translate(scale, translate))

    def union(self, other):
        """
        Return the union of this voxel grid with another.

        The grids must be axis-aligned, share the same uniform pitch and
        have origins that differ by an integer number of voxel cells, i.e.
        their lattices must be coincident in world space. Occupied cells at
        the same world location are merged so they are only counted once.

        Neither input grid is modified.

        Parameters
        ----------
        other : trimesh.voxel.VoxelGrid
          The voxel grid to union with this one.

        Returns
        ----------
        union : trimesh.voxel.VoxelGrid
          A new grid whose occupied cells are the union of the two inputs.
          Its transform places index ``[0, 0, 0]`` at the minimum occupied
          lattice corner, its shape spans every occupied cell and its
          occupied points match those of the inputs in world coordinates.

        Raises
        ----------
        ValueError
          If `other` is not a `VoxelGrid`, if either grid is rotated or
          sheared, if their pitches differ, or if their origins are not
          separated by an integer number of cells.
        """
        if not isinstance(other, VoxelGrid):
            raise ValueError(f"union requires a VoxelGrid, got {type(other).__name__}")

        def _axis_aligned(matrix):
            # the linear part must be a diagonal (no rotation/shear),
            # positive (no reflection) scaling matrix
            linear = matrix[:3, :3]
            diagonal = np.diag(linear)
            if not np.allclose(
                linear, diagonal * np.eye(3), np.abs(diagonal) * 1e-6 + 1e-8
            ) or np.any(diagonal <= 0.0):
                raise ValueError(
                    "union requires axis-aligned grids (no rotation or shear)"
                )
            return diagonal

        pitch_self = _axis_aligned(self.transform)
        pitch_other = _axis_aligned(other.transform)
        if not np.allclose(
            pitch_self, pitch_other, np.max(np.abs(pitch_self)) * 1e-6 + 1e-8
        ):
            raise ValueError(f"grid pitches differ: {pitch_self} vs {pitch_other}")
        pitch = pitch_self

        # translation of each grid's [0, 0, 0] cell in world coordinates
        origin_self = self.transform[:3, 3]
        origin_other = other.transform[:3, 3]
        # offset of `other` relative to self, measured in voxel cells
        offset = (origin_other - origin_self) / pitch
        offset_int = np.round(offset).astype(int)
        if not np.allclose(offset, offset_int, atol=1e-6, rtol=0.0):
            raise ValueError(f"origins must differ by integer cells, got {offset}")

        # empty grids contribute nothing: return a copy so the inputs
        # are never mutated or referenced by the returned grid
        if self.is_empty:
            return other.copy()
        if other.is_empty:
            return self.copy()

        # express every occupied cell in a shared lattice whose origin
        # is the minimum occupied cell corner in world coordinates
        indices_self = self.sparse_indices
        indices_other = other.sparse_indices + offset_int
        merged = np.vstack((indices_self, indices_other))
        minimum = merged.min(axis=0)
        merged -= minimum

        # drop duplicate cells so overlapping voxels count once
        merged = np.unique(merged, axis=0)
        shape = tuple((merged.max(axis=0) + 1).tolist())

        if isinstance(self.encoding, DenseEncoding) and isinstance(
            other.encoding, DenseEncoding
        ):
            dense = np.zeros(shape, dtype=bool)
            dense[tuple(merged.T)] = True
            encoding = DenseEncoding(dense)
        else:
            encoding = SparseBinaryEncoding(merged, shape=shape)

        origin = origin_self + minimum * pitch
        transform = tr.scale_and_translate(scale=pitch, translate=origin)
        return VoxelGrid(encoding, transform=transform)

    def __add__(self, other):
        if not isinstance(other, VoxelGrid):
            return NotImplemented
        return self.union(other)
