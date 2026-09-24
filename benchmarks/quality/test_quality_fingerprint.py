"""Fingerprint contract regression tests; no inference or model weights needed.

Run with: python -m unittest benchmarks.quality.test_quality_fingerprint
"""
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.quality.grade_quality import ROOT, ExpectedTask, Protocol, grade
from benchmarks.quality.quality_runner import suite_sha256


class QualityFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol: Protocol = json.loads((ROOT / 'quality_protocol.json').read_text())
        self.expected: dict[str, ExpectedTask] = json.loads((ROOT / 'quality_expected.json').read_text())

    def grade_fingerprint(self, fingerprint: str, protocol: Protocol) -> None:
        # Missing predictions are graded as errors, allowing the full grading
        # path to run without inference while exercising the fingerprint gate.
        raw = {
            'suite_sha256': fingerprint,
            'model': 'fingerprint-test',
            'results': [
                {'id': tid, 'family': task['family'], 'output': {}}
                for tid, task in self.expected.items()
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps(raw))
            rows, _ = grade(str(path), self.expected, protocol)
        self.assertEqual(len(rows), protocol['expected_decisions'])

    def test_current_runner_fingerprint(self) -> None:
        fingerprint = suite_sha256()
        self.assertEqual(fingerprint, self.protocol['suite_sha256'])
        self.grade_fingerprint(fingerprint, self.protocol)

    def test_original_results_remain_gradable(self) -> None:
        self.grade_fingerprint(
            '877e009f6d24b4d84724b5160a2fb7afd52b57b74b872d6324dda334bc0f5847',
            self.protocol,
        )

    def test_unknown_fingerprint_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, 'Unrecognized quality suite fingerprint'):
            self.grade_fingerprint('0' * 64, self.protocol)

    def test_protocol_without_compatibility_entries(self) -> None:
        protocol = self.protocol.copy()
        protocol.pop('compatible_suite_sha256', None)
        self.grade_fingerprint(protocol['suite_sha256'], protocol)
        with self.assertRaisesRegex(AssertionError, 'Unrecognized quality suite fingerprint'):
            self.grade_fingerprint(
                '877e009f6d24b4d84724b5160a2fb7afd52b57b74b872d6324dda334bc0f5847',
                protocol,
            )


if __name__ == '__main__':
    unittest.main()
