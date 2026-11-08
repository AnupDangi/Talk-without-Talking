import unittest

from device_discovery import enumerate_audio_devices, input_devices


class FakeDefaults:
    device = (1, 2)


class FakeSoundDevice:
    default = FakeDefaults()

    @staticmethod
    def query_devices():
        return [
            {
                "name": "Built-in Microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "USB Headset",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Display Audio",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]

    @staticmethod
    def query_hostapis():
        return [{"name": "Core Audio"}, {"name": "USB Audio"}]


class DeviceDiscoveryTests(unittest.TestCase):
    def test_enumerates_input_output_and_full_duplex_endpoints(self):
        endpoints = enumerate_audio_devices(FakeSoundDevice)

        self.assertEqual([item["name"] for item in endpoints], [
            "Built-in Microphone", "USB Headset", "Display Audio"
        ])
        self.assertEqual((endpoints[1]["max_input_channels"], endpoints[1]["max_output_channels"]), (1, 2))
        self.assertEqual(endpoints[0]["host_api"], "Core Audio")
        self.assertEqual(endpoints[1]["host_api"], "USB Audio")

    def test_keeps_runtime_indices_distinct_from_unavailable_stable_ids(self):
        endpoint = enumerate_audio_devices(FakeSoundDevice)[1]

        self.assertEqual(endpoint["index"], 1)
        self.assertIsNone(endpoint["stable_id"])
        self.assertFalse(endpoint["stable_id_available"])

    def test_reports_input_and_output_defaults_independently(self):
        endpoints = enumerate_audio_devices(FakeSoundDevice)

        self.assertTrue(endpoints[1]["default_input"])
        self.assertTrue(endpoints[2]["default_output"])
        self.assertFalse(endpoints[0]["default_output"])

    def test_input_compatibility_view_preserves_legacy_fields(self):
        inputs = input_devices(FakeSoundDevice)

        self.assertEqual([item["index"] for item in inputs], [0, 1])
        self.assertEqual(inputs[0]["channels"], 1)
        self.assertEqual(inputs[0]["default_samplerate"], 48000)
        self.assertFalse(inputs[0]["default"])
        self.assertTrue(inputs[1]["default"])


if __name__ == "__main__":
    unittest.main()
