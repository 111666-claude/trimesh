try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


def _assert_row_invariants(test, mesh, adjacency, edges, edge_index):
    """Shared checks for any non-manifold adjacency result."""
    adjacency = np.asarray(adjacency)
    edges = np.asarray(edges)
    edge_index = np.asarray(edge_index)
    faces = np.asarray(mesh.faces)

    # every array is row aligned
    test.assertEqual(len(adjacency), len(edges))
    test.assertEqual(len(adjacency), len(edge_index))

    if len(adjacency) == 0:
        return

    # no self adjacency
    test.assertTrue((adjacency[:, 0] != adjacency[:, 1]).all())
    # face pairs are unordered / sorted
    test.assertTrue((adjacency[:, 0] < adjacency[:, 1]).all())

    # no duplicate pair for the same edge index
    key = np.column_stack((edge_index, adjacency))
    test.assertEqual(len(np.unique(key, axis=0)), len(key))

    # every returned edge is a real edge of both paired faces
    face_sets = [set(f.tolist()) for f in faces]
    for (a, b), edge in zip(adjacency, edges):
        edge_set = set(edge.tolist())
        test.assertTrue(edge_set.issubset(face_sets[int(a)]))
        test.assertTrue(edge_set.issubset(face_sets[int(b)]))

    # edge index maps to the same unique edge row
    test.assertTrue(np.array_equal(mesh.edges_unique[edge_index], edges))


class NonManifoldAdjacencyTest(g.unittest.TestCase):
    def test_three_faces_one_edge(self):
        # fixed mesh: all three triangles share the (0, 1) edge,
        # every other edge is used by a single face
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # the default manifold interface drops the count-three edge
        default = g.trimesh.graph.face_adjacency(mesh=mesh)
        self.assertEqual(default.shape, (0, 2))

        # the non-manifold result has the C(3, 2) == 3 pairs
        adjacency = g.trimesh.graph.face_adjacency_non_manifold(mesh=mesh)
        expected = {tuple(sorted(p)) for p in [(0, 1), (0, 2), (1, 2)]}
        self.assertEqual(len(adjacency), 3)
        self.assertEqual({tuple(r) for r in adjacency}, expected)

        # query pairs and corresponding edges / edge index together
        (
            adjacency,
            edges,
            edge_index,
        ) = g.trimesh.graph.face_adjacency_non_manifold(
            mesh=mesh, return_edge_indices=True
        )
        self.assertEqual(len(adjacency), 3)
        # all three pairs correspond to the same shared edge
        self.assertTrue((edges == [0, 1]).all())
        self.assertEqual(len(np.unique(edge_index)), 1)
        _assert_row_invariants(self, mesh, adjacency, edges, edge_index)

        # the opt-in flag on the existing function returns the same
        flagged, flagged_edges = g.trimesh.graph.face_adjacency(
            mesh=mesh, non_manifold=True, return_edges=True
        )
        self.assertTrue(np.array_equal(flagged, adjacency))
        self.assertTrue(np.array_equal(flagged_edges, edges))

        # cached mesh properties agree
        self.assertTrue(np.array_equal(mesh.face_adjacency_non_manifold, adjacency))
        self.assertTrue(np.array_equal(mesh.face_adjacency_non_manifold_edges, edges))
        self.assertTrue(
            np.array_equal(mesh.face_adjacency_non_manifold_edge_index, edge_index)
        )
        # word-order mesh alias returns the same cached adjacency
        self.assertTrue(np.array_equal(mesh.non_manifold_face_adjacency, adjacency))
        # module-level word-order function alias
        self.assertIs(
            g.trimesh.graph.non_manifold_face_adjacency,
            g.trimesh.graph.face_adjacency_non_manifold,
        )

    def test_more_than_three_faces(self):
        # k faces on one edge must yield C(k, 2) pairs
        for k in (4, 5, 6):
            vertices = np.vstack(
                (
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    np.column_stack(
                        (
                            np.cos(np.linspace(0.0, 3.0, k)),
                            np.sin(np.linspace(0.0, 3.0, k)),
                            np.linspace(0.0, 1.0, k),
                        )
                    ),
                )
            )
            faces = np.array([[0, 1, i] for i in range(2, 2 + k)])
            mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            (
                adjacency,
                edges,
                edge_index,
            ) = g.trimesh.graph.face_adjacency_non_manifold(
                mesh=mesh, return_edge_indices=True
            )
            self.assertEqual(len(adjacency), k * (k - 1) // 2)
            self.assertTrue((edges == [0, 1]).all())
            _assert_row_invariants(self, mesh, adjacency, edges, edge_index)

    def test_faces_argument_path(self):
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        adjacency, edges, edge_index = g.trimesh.graph.face_adjacency_non_manifold(
            faces=faces, return_edge_indices=True
        )
        self.assertEqual(len(adjacency), 3)
        self.assertTrue((edges == [0, 1]).all())
        # edge index follows the same unique-row ordering used internally
        from trimesh.grouping import unique_rows

        edges_sorted = np.sort(g.trimesh.geometry.faces_to_edges(faces), axis=1)
        unique_idx, _ = unique_rows(edges_sorted)
        unique_edges = edges_sorted[unique_idx]
        self.assertTrue(np.array_equal(unique_edges[edge_index], edges))

    def test_degenerate_faces_no_self_adjacency(self):
        # face 0 is degenerate: [0, 1, 1] contributes edge (0, 1) twice
        faces = np.array([[0, 1, 1], [0, 1, 2], [1, 2, 3]])
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        (
            adjacency,
            edges,
            edge_index,
        ) = g.trimesh.graph.face_adjacency_non_manifold(
            mesh=mesh, return_edge_indices=True
        )
        # no self adjacency despite the repeated edge
        self.assertTrue((adjacency[:, 0] != adjacency[:, 1]).all())
        # (0, 1) still links face 0 and face 1 even though its
        # occurrence count is three due to degeneracy
        self.assertIn((0, 1), {tuple(r) for r in adjacency})
        _assert_row_invariants(self, mesh, adjacency, edges, edge_index)

        # mutating a normal mesh face to be degenerate keeps the
        # no-self-adjacency guarantee through both interfaces
        m = g.get_mesh("featuretype.STL")
        m.faces[0][2] = m.faces[0][0]
        self.assertTrue(g.np.not_equal(*m.face_adjacency.T).all())
        nm = m.face_adjacency_non_manifold
        self.assertTrue((nm[:, 0] != nm[:, 1]).all())

    def test_duplicate_triangles_keep_edge_correspondence(self):
        # faces 0 and 1 are identical duplicate triangles; they share
        # three edges and therefore appear once per shared edge
        faces = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 3]])
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        (
            adjacency,
            edges,
            edge_index,
        ) = g.trimesh.graph.face_adjacency_non_manifold(
            mesh=mesh, return_edge_indices=True
        )
        _assert_row_invariants(self, mesh, adjacency, edges, edge_index)

        pair_01 = adjacency[(adjacency == [0, 1]).all(axis=1)]
        # pair (0, 1) is returned once for each of its three edges
        self.assertEqual(len(pair_01), 3)
        self.assertEqual(
            len(np.unique(edge_index[(adjacency == [0, 1]).all(axis=1)])),
            3,
        )

    def test_double_sided_square_issue_1545(self):
        # four faces where the diagonal edge is used by all four faces
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 119.24, 0.0],
                [118.11, 0.0, 0.0],
                [118.11, 119.24, 0.0],
            ]
        )
        faces = np.array([[0, 3, 2], [3, 0, 1], [2, 3, 0], [1, 0, 3]])
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # historical default behavior is intentionally unchanged:
        # the four-way diagonal is omitted and duplicate pairs remain
        expected_default = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
        self.assertTrue(np.array_equal(mesh.face_adjacency, expected_default))

        (
            adjacency,
            edges,
            edge_index,
        ) = g.trimesh.graph.face_adjacency_non_manifold(
            mesh=mesh, return_edge_indices=True
        )
        _assert_row_invariants(self, mesh, adjacency, edges, edge_index)
        # four faces on the diagonal (zero-based vertices 0 and 3,
        # OBJ vertices 1 and 4) contribute C(4, 2) == 6 pairs
        diagonal = np.sort([0, 3])
        on_diagonal = (edges == diagonal).all(axis=1)
        self.assertEqual(on_diagonal.sum(), 6)

    def test_manifold_mesh_matches_default_pairs(self):
        # on a normal manifold mesh the complete result contains the
        # same face-pair set as the default interface
        for mesh in [
            g.trimesh.creation.box(),
            g.trimesh.creation.icosphere(subdivisions=2),
            g.get_mesh("featuretype.STL"),
        ]:
            default = {tuple(sorted(r)) for r in mesh.face_adjacency}
            non_manifold = {tuple(r) for r in mesh.face_adjacency_non_manifold}
            self.assertEqual(non_manifold, default)

            adjacency, edges, edge_index = g.trimesh.graph.face_adjacency_non_manifold(
                mesh=mesh, return_edge_indices=True
            )
            _assert_row_invariants(self, mesh, adjacency, edges, edge_index)

    def test_empty_inputs(self):
        empty_faces = np.zeros((0, 3), dtype=np.int64)
        adjacency = g.trimesh.graph.face_adjacency_non_manifold(faces=empty_faces)
        self.assertEqual(adjacency.shape, (0, 2))
        adjacency, edges, edge_index = g.trimesh.graph.face_adjacency_non_manifold(
            faces=empty_faces, return_edge_indices=True
        )
        self.assertEqual(adjacency.shape, (0, 2))
        self.assertEqual(edges.shape, (0, 2))
        self.assertEqual(edge_index.shape, (0,))

    def test_kwarg_aliases(self):
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        canonical = g.trimesh.graph.face_adjacency(faces=faces, non_manifold=True)
        for alias in (
            "include_non_manifold",
            "allow_non_manifold",
            "nonmanifold",
        ):
            result = g.trimesh.graph.face_adjacency(faces=faces, **{alias: True})
            self.assertTrue(np.array_equal(result, canonical))
        with self.assertRaises(TypeError):
            g.trimesh.graph.face_adjacency(faces=faces, bogus=True)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
