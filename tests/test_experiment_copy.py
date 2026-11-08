from pathlib import Path
import unittest


class ExperimentCopyTests(unittest.TestCase):
    def test_jawlab_explains_labels_and_audio_only_comparison(self):
        html = (Path(__file__).parents[1] / "static" / "experiment.html").read_text()

        self.assertIn("Condition and word are labels you assign to each take", html)
        self.assertIn("they are not recognized outputs", html)
        self.assertIn("JawLab computes signal metrics from audio recorded by the selected microphone", html)
        self.assertIn("This page does not decode arbitrary silently mouthed words", html)
        self.assertIn("Word label (your prompt)", html)
        self.assertIn("Condition label", html)
        self.assertIn("Word label", html)
        self.assertIn("Microphone signal monitor", html)
        self.assertIn("No raw audio is saved by this logger", html)
        self.assertIn("Frequency bands · recent signal energy", html)


if __name__ == "__main__":
    unittest.main()
