import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from word_alignment import gemma_selection


class SplitRetryTest(unittest.TestCase):
    def test_missing_decision_splits_only_failed_batch_and_is_resumable(self):
        calls = []
        interrupt = True

        class Model:
            revision = "test-revision"

            def __init__(self, args, revision):
                pass

            def generate(self, messages):
                payload = json.JSONDecoder().raw_decode(messages[1]["content"])[0]
                ids = tuple(row["line_id"] for row in payload["sentences"])
                calls.append(ids)
                if ids == (3, 4) and interrupt:
                    raise RuntimeError("generation interrupted")
                returned = [i for i in ids if i != 4 or ids == (4,)]
                # Force the irrecoverable single-line case independently of the first batch.
                returned = [i for i in returned if i != 6]
                return json.dumps({"decisions": [{"line_id": i, "take": ["c1"]} for i in returned]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "details").mkdir()
            source = root / "candidates.jsonl"
            method = "C_nltk_treebank"
            rows = []
            for i in range(1, 9):
                baseline = {"pair_id": f"old{i}", "ko": "병", "en": "pain", "ko_span": [0, 1], "en_span": [0, 4]}
                pair = {"pair_id": f"new{i}", "ko": "병명", "en": "painful", "ko_span": [0, 2], "en_span": [0, 7],
                        "score": 0.7, "evidence": {"margin": 0.2}}
                rows.append({"line_id": i, "method": method, "ko": "병명", "en": "painful",
                             "baseline_pairs": [baseline], "candidates": [{"pair": pair, "en_to_ko_rank": 1, "ko_to_en_rank": 2}],
                             "source_status": "aligned", "term_diagnostics": {}, "analysis_signature": "test"})
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan = {"settings": {"min_similarity": 0.5, "min_margin": 0.03, "max_candidates": 3,
                                 "batch_size": 4, "revision": "main"},
                    "sources": {method: str(source)}, "primary": method, "limit": 0}

            with patch.object(gemma_selection, "Generator", Model):
                with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
                    gemma_selection.select_terms(root, plan)
                self.assertFalse((root / "term-pairs.jsonl").exists())
                interrupt = False
                gemma_selection.select_terms(root, plan)

            self.assertEqual(calls.count((1, 2)), 1)
            self.assertEqual(calls.count((1, 2, 3, 4)), 3)
            self.assertEqual(calls.count((4,)), 1)
            self.assertEqual(calls.count((7, 8)), 1)
            self.assertIn((6,), calls)
            self.assertNotIn((7,), calls)
            result = root / "term-pairs.jsonl"
            published = [json.loads(line) for line in result.read_text().splitlines()]
            self.assertEqual([r["pairs"][0]["pair_id"] for r in published],
                             [f"{'old' if i == 6 else 'new'}{i}" for i in range(1, 9)])
            audit = [json.loads(line) for line in (root / "details/gemma-decisions.jsonl").read_text().splitlines()]
            self.assertIn("fallback_error", audit[5])
            summary = json.loads((root / "details/gemma-summary.json").read_text())["methods"][method]
            self.assertEqual(summary["fallback_sentences"], 1)

            saved = result.read_bytes()
            with patch.object(gemma_selection, "Generator", side_effect=AssertionError("model should not reload")):
                gemma_selection.select_terms(root, plan)
            self.assertEqual(result.read_bytes(), saved)
