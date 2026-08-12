from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.diagnose_robo2vlm_invalid_row import repair_jsonl
except ModuleNotFoundError:
    from diagnose_robo2vlm_invalid_row import repair_jsonl


class Robo2VLMDiagnosticRepairTest(unittest.TestCase):
    def test_repair_inserts_missing_source_index_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jsonl_path = Path(temporary) / "robo2vlm_train.jsonl"
            existing = [
                {"id": "q0", "images": ["/images/000000000_q0.png"]},
                {"id": "q2", "images": ["/images/000000002_q2.png"]},
            ]
            jsonl_path.write_text(
                "".join(json.dumps(record) + "\n" for record in existing), encoding="utf-8"
            )
            inserted = {"id": "q1", "images": ["/images/000000001_q1.png"]}

            repair_jsonl(jsonl_path, 1, inserted, expected_rows=3)

            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["id"] for record in records], ["q0", "q1", "q2"])
            self.assertFalse(jsonl_path.with_name(f"{jsonl_path.name}.repair.tmp").exists())


if __name__ == "__main__":
    unittest.main()
