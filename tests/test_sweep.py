import numpy as np
from pyinstrument import Profiler
from shapely.geometry import Point, Polygon

import trimesh
from trimesh.creation import sweep_polygon

trimesh.tol.strict = True


def arc_2d(final_angle, d=1, start_angle=0, splits=20):
    xs = np.linspace(start_angle, final_angle, splits)
    return np.stack((d / 2 * np.cos(xs), d / 2 * np.sin(xs)), axis=1)


def test_simple_closed(h1=1, w=2, r=4.4):
    # a simple closed path
    square = Polygon([(-0.01, -h1 / 2), (w, -h1 / 2), (w, +h1 / 2), (-0.01, +h1 / 2)])
    circle = Point([0, 0]).buffer(1.0)

    # a simple square of side `r`
    path = [[0, 0, 0], [r, 0, 0], [r, r, 0], [0, r, 0], [0, 0, 0]]

    a = sweep_polygon(square, path)
    assert a.is_volume

    aa = sweep_polygon(square, path, connect=False)
    assert aa.is_volume

    # should have the same bounds but no longer a volume
    aaa = sweep_polygon(square, path, connect=False, cap=False)
    assert np.allclose(aa.bounds, aaa.bounds)
    assert not aaa.is_volume

    b = sweep_polygon(circle, path)
    assert b.is_volume


def test_simple_extrude(height=10):
    # make sure sweep behaves okay on a simple single segment path
    # this should be identical to the extrude operation
    circle = Point([0, 0]).buffer(1.0)

    # a simple extrusion
    path = [[0, 0, 0], [0, 0, height]]
    # will be running asserts inside function
    b = sweep_polygon(circle, path)

    # should be a straight extrude along Z
    expected = np.append(np.ptp(np.reshape(circle.bounds, (2, 2)), axis=0), height)
    assert np.allclose(expected, b.extents)

    # should be well constructed
    assert b.is_volume
    # volume should correspond to expected cylinder area
    assert np.isclose(b.volume, circle.area * height)


def test_simple_open():
    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


def test_spline_3D():
    circle = Point([0, 0]).buffer(1.0)
    # make a 3D noodle using spline smoothing
    path = trimesh.path.simplify.resample_spline(
        [[0, 0, 0], [4, 4, 0], [5, 5, 0], [10, 0, 10], [0, 20, 0]], smooth=0.25, count=100
    )

    a = sweep_polygon(circle, path)
    assert a.is_volume
    assert a.body_count == 1


def test_screw():
    h = 200
    d = 8
    h1 = 1
    w = 2
    d = d - w
    h2 = h1

    lead = 2
    splits_per_turn = 4 * 20
    polygon = np.array([(-0.01, -h1 / 2), (w, -h2 / 2), (w, +h2 / 2), (-0.01, +h1 / 2)])
    h_ = h + h1
    n = int(np.ceil(splits_per_turn * h_ / lead))
    turns = n / splits_per_turn
    xys = arc_2d(2 * np.pi * turns, d, splits=n)
    zs = np.linspace(0, h_, n)[:, None]
    spin_path = np.concatenate((xys, zs), axis=1)

    with Profiler() as P:
        x = sweep_polygon(Polygon(polygon), spin_path)
    P.print()

    assert x.is_volume
    # should have produced a result corresponding to the input dimensions
    # note that the cross-section is parallel transported along the path
    # so on a long helix it rotates relative to the world axial plane and
    # the extents may grow up to the sweep of the furthest profile corner
    reach = np.hypot(w, h1 / 2.0)
    assert np.all(x.extents[:2] > d + w * 2 - 0.01)
    assert np.all(x.extents[:2] < d + 2 * reach + 0.01)
    assert x.extents[2] > h + h1 - 0.01
    assert x.extents[2] < h + h1 + 2 * reach + 0.01

    # check without capping: should be the same size just without caps
    o = sweep_polygon(Polygon(polygon), spin_path, cap=False, connect=False)
    assert np.allclose(x.bounds, o.bounds)
    assert not o.is_volume

    # try spinning along an angle
    sweep_polygon(
        Polygon(polygon), spin_path, angles=np.linspace(0.0, np.pi / 2, len(spin_path))
    )

    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


def test_curved_path_low_twist():
    # a fixed seven-point curved path through 3D
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.4],
            [2.0, 1.0, 0.6],
            [3.0, 2.2, 0.1],
            [4.0, 2.6, -0.6],
            [5.0, 3.4, 0.3],
            [6.0, 3.8, 1.0],
        ]
    )
    # a regular pentagon cross-section
    angle = 2.0 * np.pi * np.arange(5) / 5.0
    pentagon = Polygon(np.column_stack([np.cos(angle), np.sin(angle)]))

    mesh = sweep_polygon(pentagon, path)
    # mesh should be a single connected watertight body
    assert mesh.is_volume
    assert mesh.body_count == 1
    assert mesh.is_watertight

    def slice_marker(m, count):
        # vertices are stored as one block of boundary
        # vertices per slice: track the direction from each
        # slice center to the first polygon vertex
        slices = m.vertices[: count * 5].reshape((count, 5, 3))
        marker = slices[:, 0] - slices.mean(axis=1)
        return marker / np.linalg.norm(marker, axis=1, keepdims=True)

    def slice_normals(p, closed=False):
        vector = trimesh.util.unitize(np.diff(p, axis=0))
        normal = np.concatenate(
            [[vector[0]], trimesh.util.unitize(vector[1:] + vector[:-1]), [vector[-1]]]
        )
        if closed:
            normal[0] = trimesh.util.unitize(normal[[0, -1]].mean(axis=0))
        return normal

    def transport(a, b):
        # minimal rotation taking unit vector `a` to `b`
        axis = np.cross(a, b)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.eye(3)
        angle = np.arctan2(norm, np.dot(a, b))
        return trimesh.transformations.rotation_matrix(angle, axis / norm)[:3, :3]

    def extra_twist(marker, normal, i, j):
        # angle between the marker of slice `j` and the marker of
        # slice `i` after parallel transport along the normals
        moved = marker[i]
        for k in range(i, j):
            moved = transport(normal[k], normal[k + 1]).dot(moved)
        if j == len(marker):  # wrap around the seam of a closed path
            moved = transport(normal[-1], normal[0]).dot(moved)
            j = 0
        return np.arccos(np.clip(np.dot(moved, marker[j]), -1.0, 1.0))

    # with default angles adjacent sections should have no twist
    # beyond the parallel transport of the frame along the path
    marker = slice_marker(mesh, len(path))
    normal = slice_normals(path)
    for i in range(len(path) - 1):
        assert extra_twist(marker, normal, i, i + 1) < 1e-6

    # explicit angles should appear exactly as the per-slice rotation
    # rather than flipping with world coordinate directions
    angles = np.linspace(0.0, 0.6, len(path))
    rolled = sweep_polygon(pentagon, path, angles=angles)
    marker = slice_marker(rolled, len(path))
    for i in range(len(path) - 1):
        twist = extra_twist(marker, normal, i, i + 1)
        assert np.isclose(twist, abs(angles[i + 1] - angles[i]))

    # degenerate adjacent points should be handled explicitly
    degraded = np.insert(path, 3, path[3], axis=0)
    assert sweep_polygon(pentagon, degraded).is_volume

    # a closed version of the path should connect into a single volume
    closed = np.vstack([path, path[:1]])
    mesh_closed = sweep_polygon(pentagon, closed)
    assert mesh_closed.is_volume
    assert mesh_closed.body_count == 1

    # the total twist around the closed loop should be distributed
    # evenly so every segment including the seam carries the same
    # small twist rather than the seam absorbing it all at once
    marker = slice_marker(mesh_closed, len(closed) - 1)
    normal = slice_normals(closed, closed=True)
    twists = [extra_twist(marker, normal, i, i + 1) for i in range(len(marker) - 1)]
    twists.append(extra_twist(marker, normal, len(marker) - 1, len(marker)))
    assert np.allclose(twists, twists[0], atol=1e-6)


if __name__ == "__main__":
    test_simple_closed()
    test_simple_extrude()
    test_simple_open()
    test_screw()
    test_spline_3D()
    test_curved_path_low_twist()
