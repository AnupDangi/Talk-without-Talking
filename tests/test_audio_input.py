import unittest
from unittest.mock import patch

import numpy as np

import server


def input_info(*, rate=48000, channels=1):
    return {
        "name": "Synthetic Microphone",
        "default_samplerate": rate,
        "max_input_channels": channels,
    }


class FakeInputStream:
    def __init__(self, *, callback, **settings):
        self.callback = callback
        self.settings = settings
        self.closed = False

    def start(self):
        if "on_start" in self.settings:
            self.settings["on_start"](self.callback)

    def stop(self):
        pass

    def close(self):
        self.closed = True


class MicrophoneFormatNegotiationTests(unittest.TestCase):
    def make_source(self, info):
        with patch.object(server, "resolve_input_device", return_value=(info, 7)):
            return server.MicSource(7)

    def test_default_mono_format_is_tried_first(self):
        source = self.make_source(input_info())
        attempted = []

        def make_stream(**settings):
            attempted.append((settings["samplerate"], settings["channels"]))
            return FakeInputStream(**settings)

        with patch.object(server.sd, "InputStream", side_effect=make_stream):
            with patch.object(server.sd, "check_input_settings") as check_settings:
                source.start()

        self.addCleanup(source.stop)
        self.assertEqual(attempted, [(48000, 1)])
        check_settings.assert_not_called()
        self.assertEqual(source.samplerate, 48000)
        self.assertEqual(source.channels, 1)

    def test_uses_validated_rate_fallback_when_default_cannot_open(self):
        source = self.make_source(input_info(rate=48000))
        attempted = []

        def make_stream(**settings):
            attempted.append((settings["samplerate"], settings["channels"]))
            if settings["samplerate"] != 16000:
                raise RuntimeError("unsupported sample rate")
            return FakeInputStream(**settings)

        def check_settings(*, samplerate, channels, **_kwargs):
            if samplerate != 16000 or channels != 1:
                raise ValueError("format unsupported")

        with patch.object(server.sd, "InputStream", side_effect=make_stream):
            with patch.object(server.sd, "check_input_settings", side_effect=check_settings):
                source.start()

        self.addCleanup(source.stop)
        self.assertEqual(attempted[0], (48000, 1))
        self.assertEqual(attempted[-1], (16000, 1))
        self.assertEqual(source.samplerate, 16000)

    def test_uses_common_rate_when_host_does_not_report_a_default(self):
        source = self.make_source(input_info(rate=None))
        attempted = []

        def make_stream(**settings):
            attempted.append((settings["samplerate"], settings["channels"]))
            return FakeInputStream(**settings)

        with patch.object(server.sd, "InputStream", side_effect=make_stream):
            with patch.object(server.sd, "check_input_settings") as check_settings:
                source.start()

        self.addCleanup(source.stop)
        self.assertEqual(attempted, [(48000, 1)])
        check_settings.assert_not_called()
        self.assertEqual(source.samplerate, 48000)

    def test_uses_stereo_if_host_rejects_mono_and_downmixes_for_processing(self):
        source = self.make_source(input_info(channels=2))
        opened = []

        def make_stream(**settings):
            opened.append(settings["channels"])
            if settings["channels"] != 2:
                raise RuntimeError("mono format unsupported")
            settings["on_start"] = lambda callback: callback(
                np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32),
                2,
                None,
                None,
            )
            return FakeInputStream(**settings)

        def check_settings(*, channels, **_kwargs):
            if channels != 2:
                raise ValueError("mono format unsupported")

        with patch.object(server.sd, "InputStream", side_effect=make_stream):
            with patch.object(server.sd, "check_input_settings", side_effect=check_settings):
                source.start()

        self.addCleanup(source.stop)
        self.assertEqual(opened, [1, 2])
        self.assertEqual(source.channels, 2)
        np.testing.assert_allclose(source.queue.get_nowait(), [0.2, 0.3])

    def test_failed_formats_report_generic_causes_and_last_portaudio_error(self):
        source = self.make_source(input_info())

        def make_stream(**_settings):
            raise RuntimeError("Internal PortAudio error [-9986]")

        def check_settings(**_settings):
            raise ValueError("unsupported settings")

        with patch.object(server.sd, "InputStream", side_effect=make_stream):
            with patch.object(server.sd, "check_input_settings", side_effect=check_settings):
                with self.assertRaisesRegex(RuntimeError, "Internal PortAudio error") as caught:
                    source.start()

        message = str(caught.exception)
        self.assertIn("microphone permission", message)
        self.assertIn("device/profile selection", message)
        self.assertIn("supported sample rate/channel formats", message)
        self.assertNotIn("Check microphone permission for the app", message)


if __name__ == "__main__":
    unittest.main()
