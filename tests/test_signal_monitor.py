import unittest

import numpy as np

from signal_monitor import LiveSignalMonitor


class LiveSignalMonitorTests(unittest.TestCase):
    def frame(self, samples, *, calibrated=True):
        monitor = LiveSignalMonitor(publish_interval_seconds=0)
        return monitor.update(
            samples,
            16000,
            device="Synthetic input",
            noise_floor_db=-60,
            calibrated=calibrated,
            force=True,
        )

    def test_silence_reports_below_floor_and_no_raw_audio(self):
        frame = self.frame(np.zeros(8192, dtype=np.float32))

        self.assertEqual(frame["activity"], "near_silence")
        self.assertLessEqual(frame["window_seconds"], 0.257)
        self.assertEqual(len(frame["bands_dbfs"]), 8)
        self.assertNotIn("samples", frame)
        self.assertNotIn("audio", frame)

    def test_periodic_tone_is_voiced_like_not_a_word_prediction(self):
        rate = 16000
        time_axis = np.arange(8192, dtype=np.float32) / rate
        tone = 0.12 * np.sin(2 * np.pi * 180 * time_axis)

        frame = self.frame(tone)

        self.assertEqual(frame["activity"], "voiced_like_activity")
        self.assertGreater(frame["voicing_score"], 0.45)
        self.assertNotIn("word", frame)
        self.assertNotIn("prediction", frame)

    def test_noise_reports_uncertain_activity_and_unready_floor_calibrates(self):
        noise = np.random.default_rng(17).normal(0, 0.05, 8192).astype(np.float32)

        frame = self.frame(noise)
        calibrating = self.frame(noise, calibrated=False)

        self.assertEqual(frame["activity"], "acoustic_activity_uncertain")
        self.assertEqual(calibrating["activity"], "calibrating")
        self.assertGreater(frame["peak_dbfs"], frame["rms_dbfs"])


if __name__ == "__main__":
    unittest.main()
