from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from universal_dataset.io import write_jsonl
from universal_dataset.split import canonical_asset_identities, split_jsonl


def _record(record_id, group_id, asset):
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": record_id,
        "group_id": group_id,
        "assets": [asset],
    }


class IdentityGraphRegressionTest(unittest.TestCase):
    def _split(self, records):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source.jsonl"
        write_jsonl(source, records)
        return split_jsonl(source, root / "train.jsonl", root / "val.jsonl", val_ratio=0.5)

    def test_transitive_alias_component_rejects_two_hashes(self):
        source_a = {"namespace": "example", "value": "A", "verified": True}
        source_b = {"namespace": "example", "value": "B", "verified": True}
        records = [
            _record(
                "bridge",
                "group-bridge",
                {
                    "id": "bridge-asset",
                    "kind": "image",
                    "uri": "/media/bridge.jpg",
                    "source_identities": [source_a, source_b],
                },
            ),
            _record(
                "left",
                "group-left",
                {
                    "id": "left-asset",
                    "kind": "image",
                    "uri": "/media/left.jpg",
                    "sha256": "a" * 64,
                    "source_identities": [source_a],
                },
            ),
            _record(
                "right",
                "group-right",
                {
                    "id": "right-asset",
                    "kind": "image",
                    "uri": "/media/right.jpg",
                    "sha256": "b" * 64,
                    "source_identities": [source_b],
                },
            ),
        ]

        with self.assertRaisesRegex(ValueError, "conflicting sha256"):
            self._split(records)

    def test_only_explicitly_verified_source_identities_connect_assets(self):
        unverified_values = [
            {"namespace": "example", "value": "shared", "verified": False},
            {"namespace": "example", "value": "shared"},
            {"namespace": "example", "value": "shared", "verified": 1},
            "not-an-object",
        ]
        input_path = Path("records.jsonl").resolve()
        identities = canonical_asset_identities(
            {
                "id": "asset",
                "uri": "/media/asset.jpg",
                "source_identities": unverified_values,
            },
            input_path,
        )
        self.assertEqual(identities, ("posix-path:/media/asset.jpg",))

        records = [
            _record(
                "first",
                "group-first",
                {
                    "id": "first-asset",
                    "kind": "image",
                    "uri": "/media/first.jpg",
                    "source_identities": [unverified_values[0]],
                },
            ),
            _record(
                "second",
                "group-second",
                {
                    "id": "second-asset",
                    "kind": "image",
                    "uri": "/media/second.jpg",
                    "source_identities": [unverified_values[1]],
                },
            ),
        ]
        summary = self._split(records)
        self.assertEqual(summary.total_components, 2)
        self.assertEqual(summary.unique_assets, 2)

    def test_file_uri_authority_is_preserved_except_for_localhost(self):
        input_path = Path("records.jsonl").resolve()

        def identity(uri):
            return canonical_asset_identities({"id": uri, "uri": uri}, input_path)[0]

        self.assertNotEqual(
            identity("file://server-a/share/image.jpg"),
            identity("file://server-b/share/image.jpg"),
        )
        self.assertEqual(
            identity("file://LOCALHOST/share/image.jpg"),
            identity("/share/image.jpg"),
        )

    def test_remote_uri_query_remains_identity_bearing_and_fragment_does_not(self):
        input_path = Path("records.jsonl").resolve()

        def identity(uri):
            return canonical_asset_identities({"id": uri, "uri": uri}, input_path)[0]

        self.assertNotEqual(
            identity("https://example.test/media?id=A"),
            identity("https://example.test/media?id=B"),
        )
        self.assertEqual(
            identity("https://example.test/media?id=A#first"),
            identity("https://example.test/media?id=A#second"),
        )

    def test_external_only_source_ref_uses_complete_selector(self):
        input_path = Path("records.jsonl").resolve()

        def identities(source_ref):
            return canonical_asset_identities(
                {"id": "external", "kind": "tensor", "source_ref": source_ref},
                input_path,
            )

        container = "https://example.test/dataset.h5?revision=1"
        self.assertNotEqual(
            identities({"uri": container, "key": "camera/front"}),
            identities({"uri": container, "key": "camera/rear"}),
        )
        self.assertNotEqual(
            identities({"uri": container, "offset": 0, "length": 1024}),
            identities({"uri": container, "offset": 1024, "length": 1024}),
        )

    def test_source_ref_hash_is_an_alias_and_detects_conflicts(self):
        selector = {
            "uri": "https://example.test/dataset.h5",
            "key": "camera/front",
        }
        same_content = [
            _record(
                "same-content-first",
                "same-content-group-first",
                {
                    "id": "same-content-first-asset",
                    "kind": "tensor",
                    "source_ref": dict(selector, sha256="a" * 64),
                },
            ),
            _record(
                "same-content-second",
                "same-content-group-second",
                {
                    "id": "same-content-second-asset",
                    "kind": "tensor",
                    "source_ref": {
                        "uri": selector["uri"],
                        "key": "camera/rear",
                        "sha256": "a" * 64,
                    },
                },
            ),
        ]
        self.assertEqual(
            canonical_asset_identities(same_content[0]["assets"][0], Path("records.jsonl"))[0],
            "sha256:" + "a" * 64,
        )
        summary = self._split(same_content)
        self.assertEqual(summary.total_components, 1)
        self.assertEqual(summary.unique_assets, 1)

        records = [
            _record(
                "first",
                "group-first",
                {
                    "id": "first-asset",
                    "kind": "tensor",
                    "source_ref": dict(selector, sha256="a" * 64),
                },
            ),
            _record(
                "second",
                "group-second",
                {
                    "id": "second-asset",
                    "kind": "tensor",
                    "source_ref": dict(selector, sha256="b" * 64),
                },
            ),
        ]

        with self.assertRaisesRegex(ValueError, "conflicting sha256"):
            self._split(records)


if __name__ == "__main__":
    unittest.main()
