"""
test_gltf_handlers.py
---------------------

Pin the glTF extension-handler interface that upstream packages register against.
"""

import base64

try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.exchange.gltf import extensions as ext


def minimal_gltf(extension="TEST_dummy"):
    # one-triangle glTF whose POSITION accessor is bufferless — a placeholder
    # an extension is expected to fill — and whose primitive carries the extension
    indices = g.np.array([0, 1, 2], dtype=g.np.uint32).tobytes()
    uri = "data:application/octet-stream;base64," + base64.b64encode(indices).decode()
    tree = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "extensions": {extension: {}},
                    }
                ]
            }
        ],
        "accessors": [
            # bufferless — created as placeholder zeros
            {"componentType": 5126, "count": 3, "type": "VEC3"},
            {"componentType": 5125, "count": 3, "type": "SCALAR", "bufferView": 0},
        ],
        "bufferViews": [{"buffer": 0, "byteLength": 12, "byteOffset": 0}],
        "buffers": [{"byteLength": 12, "uri": uri}],
    }
    return g.io.BytesIO(g.json.dumps(tree).encode())


class GLTFHandlerTest(g.unittest.TestCase):
    def setUp(self):
        # snapshot the global handler registry so tests don't leak
        self._snapshot = {scope: dict(h) for scope, h in ext._handlers.items()}

    def tearDown(self):
        ext._handlers.clear()
        ext._handlers.update(self._snapshot)

    def test_register_roundtrip(self):
        @ext.register_handler("TEST_x", scope="primitive_preprocess")
        def handler(context):
            return None

        # the decorator registers into the scope registry and returns the function
        assert ext._handlers["primitive_preprocess"]["TEST_x"] is handler

    def test_preprocess_context_keys(self):
        # the loader must pass the documented context keys to preprocess handlers
        seen = {}

        @ext.register_handler("TEST_dummy", scope="primitive_preprocess")
        def spy(context):
            seen.update({k: type(v) for k, v in context.items()})
            return None

        g.trimesh.load(minimal_gltf(), file_type="gltf")
        assert {"data", "primitive", "accessors", "views"}.issubset(seen)

    def test_no_handler_warns_and_zeros(self):
        # without a handler the placeholder geometry stays zero and we warn by name
        with self.assertLogs("trimesh", level="WARNING") as cm:
            scene = g.trimesh.load(minimal_gltf(), file_type="gltf")
        geom = next(iter(scene.geometry.values()))
        assert g.np.allclose(geom.vertices, 0.0)
        assert any("TEST_dummy" in line for line in cm.output)

    def test_draco_warns_by_name(self):
        # a draco-compressed primitive with no registered decoder must
        # warn naming the real extension
        with self.assertLogs("trimesh", level="WARNING") as cm:
            g.trimesh.load(minimal_gltf("KHR_draco_mesh_compression"), file_type="gltf")
        assert any("KHR_draco_mesh_compression" in line for line in cm.output)

    def test_handler_fills_placeholder(self):
        # a handler may fill the placeholder accessor in place — a successful
        # decode must not warn
        filled = g.np.arange(9, dtype=g.np.float32).reshape((3, 3))

        @ext.register_handler("TEST_dummy", scope="primitive_preprocess")
        def fill(context):
            context["accessors"][context["primitive"]["attributes"]["POSITION"]][:] = (
                filled
            )
            return None

        with self.assertNoLogs("trimesh", level="WARNING"):
            scene = g.trimesh.load(minimal_gltf(), file_type="gltf")
        geom = next(iter(scene.geometry.values()))
        assert g.np.allclose(geom.vertices, filled)


class TextureTransformTest(g.unittest.TestCase):
    """KHR_texture_transform parsing and the spec UV matrix."""

    def test_no_extension_returns_none(self):
        # plain textureInfo objects must be left on the old code path
        assert ext.parse_texture_transform({"index": 0}) is None
        assert ext.parse_texture_transform({}) is None
        assert ext.parse_texture_transform(None) is None

    def test_defaults_are_identity(self):
        # an empty extension object applies the spec defaults: no movement
        transform = ext.parse_texture_transform(
            {"extensions": {"KHR_texture_transform": {}}}
        )
        assert g.np.allclose(transform["offset"], [0.0, 0.0])
        assert g.np.allclose(transform["scale"], [1.0, 1.0])
        assert transform["rotation"] == 0.0
        assert transform["tex_coord"] == 0

        uv = g.np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        assert g.np.allclose(ext.apply_texture_transform(uv, transform), uv)

    def test_offset_u(self):
        # the Khronos TextureTransformTest Offset U case: the first UV's
        # U component moves from 0 to 0.5
        transform = ext.parse_texture_transform(
            {"extensions": {"KHR_texture_transform": {"offset": [0.5, 0.0]}}}
        )
        uv = g.np.array([[0.0, 0.0], [1.0, 1.0]])
        result = ext.apply_texture_transform(uv, transform)
        assert g.np.allclose(result[0], [0.5, 0.0])
        assert g.np.allclose(result[1], [1.5, 1.0])
        # the input array must never be mutated
        assert g.np.allclose(uv, g.np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_combined_matches_spec_formula(self):
        # u' = ou + cos*su*u - sin*sv*v ; v' = ov + sin*su*u + cos*sv*v
        transform = ext.parse_texture_transform(
            {
                "extensions": {
                    "KHR_texture_transform": {
                        "offset": [-0.2, -0.1],
                        "rotation": 0.3,
                        "scale": [1.5, 1.5],
                    }
                }
            }
        )
        uv = g.np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        offset, rotation, scale = (
            transform["offset"],
            transform["rotation"],
            transform["scale"],
        )
        c, s = g.np.cos(rotation), g.np.sin(rotation)
        expected = g.np.column_stack(
            [
                offset[0] + c * scale[0] * uv[:, 0] - s * scale[1] * uv[:, 1],
                offset[1] + s * scale[0] * uv[:, 0] + c * scale[1] * uv[:, 1],
            ]
        )
        assert g.np.allclose(ext.apply_texture_transform(uv, transform), expected)

    def test_tex_coord_override(self):
        # the extension's texCoord shadows the textureInfo-level one
        transform = ext.parse_texture_transform(
            {
                "texCoord": 0,
                "extensions": {"KHR_texture_transform": {"texCoord": 1}},
            }
        )
        assert transform["tex_coord"] == 1

        # without an extension-level value the textureInfo default is used
        transform = ext.parse_texture_transform(
            {"texCoord": 2, "extensions": {"KHR_texture_transform": {}}}
        )
        assert transform["tex_coord"] == 2

    def test_non_finite_values_rejected(self):
        for bad in (
            {"offset": [g.np.nan, 0.0]},
            {"scale": [1.0, g.np.inf]},
            {"rotation": g.np.inf},
        ):
            assert (
                ext.parse_texture_transform(
                    {"extensions": {"KHR_texture_transform": bad}}
                )
                is None
            )

    def test_illegal_tex_coord_rejected(self):
        assert (
            ext.parse_texture_transform(
                {"extensions": {"KHR_texture_transform": {"texCoord": -1}}}
            )
            is None
        )
        assert (
            ext.parse_texture_transform(
                {"extensions": {"KHR_texture_transform": {"texCoord": "nope"}}}
            )
            is None
        )

    def test_malformed_values_rejected(self):
        # wrong vector lengths, nested junk and non-numeric scalars must all
        # degrade to "no transform" rather than raising into the loader
        for bad in (
            {"offset": [1.0]},
            {"offset": [1.0, 2.0, 3.0]},
            {"scale": [[1.0], [1.0]]},
            {"rotation": "wide"},
            {"offset": "nope"},
        ):
            assert (
                ext.parse_texture_transform(
                    {"extensions": {"KHR_texture_transform": bad}}
                )
                is None
            )

        # a malformed extension entry must be ignored, not crash parsing
        assert (
            ext.parse_texture_transform({"extensions": {"KHR_texture_transform": "junk"}})
            is None
        )
        assert (
            ext.parse_texture_transform({"extensions": {"other_extension": {}}}) is None
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
