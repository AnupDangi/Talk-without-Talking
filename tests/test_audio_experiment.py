import tempfile
import unittest

import numpy as np

from audio_experiment import AudioTrialRecorder


class AudioTrialRecorderTests(unittest.TestCase):
    def make_recorder(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return AudioTrialRecorder(temp_dir.name)

    def record(self, recorder, condition, word, tone_hz=150):
        trial_id = recorder.start_trial(
            {"index": 4, "name": "Synthetic Input", "default_samplerate": 16000},
            condition=condition,
            word=word,
            duration_sec=2,
        )["trial_id"]
        rate = 16000
        t = np.arange(rate, dtype=np.float32) / rate
        signal = (0.15 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
        recorder.push_block(signal, rate)
        return recorder.stop_trial(trial_id)

    def test_trial_requires_active_capture_and_silent_word_label(self):
        recorder = self.make_recorder()

        with self.assertRaisesRegex(ValueError, "select an input"):
            recorder.start_trial(None, condition="still", word=None, duration_sec=2)
        with self.assertRaisesRegex(ValueError, "word label"):
            recorder.start_trial(
                {"index": 0, "name": "Mic"},
                condition="silent",
                word=None,
                duration_sec=2,
            )

    def test_trial_saves_metrics_and_prevents_overlapping_recordings(self):
        recorder = self.make_recorder()
        first = recorder.start_trial(
            {"index": 4, "name": "Synthetic Input", "default_samplerate": 16000},
            condition="jaw_open",
            word=None,
            duration_sec=2,
        )
        with self.assertRaisesRegex(RuntimeError, "already active"):
            recorder.start_trial(
                {"index": 4, "name": "Synthetic Input"},
                condition="still",
                word=None,
                duration_sec=2,
            )

        recorder.push_block(np.zeros(16000, dtype=np.float32), 16000)
        result = recorder.stop_trial(first["trial_id"])

        self.assertEqual(result["trial"]["condition"], "jaw_open")
        self.assertEqual(result["trial"]["device"], "Synthetic Input")
        self.assertLess(result["trial"]["metrics"]["rms_db"], 0)
        self.assertIn("low_rel_db", result["trial"]["metrics"])
        self.assertEqual(recorder.state()["trial_count"], 1)

    def test_silent_word_metrics_report_labeled_accuracy_and_chance(self):
        recorder = self.make_recorder()
        for word, tone_hz in [("yes", 120), ("no", 260)]:
            for _ in range(3):
                self.record(recorder, "silent", word, tone_hz)

        report = recorder.state()["report"]

        self.assertIsNotNone(report)
        self.assertEqual(report["word_chance"], 0.5)
        self.assertIsInstance(report["word_accuracy"], float)
        self.assertIn("word_confusion", report)


if __name__ == "__main__":
    unittest.main()
