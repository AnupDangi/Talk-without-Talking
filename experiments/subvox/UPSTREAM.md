# Upstream source and local changes

Subvox is the MIT-licensed project by Joe Blau, retained here with its `LICENSE`.

- Upstream: https://github.com/joeblau/subvox
- Imported from `main` at commit `591d02838521c3d33107bc9fde1a6fd620f96d58`
- Local changes: the CoreMotion status text and sensor-screen copy now describe the runtime capability result generically and point to JawLab when motion is unsupported. Decode also reports a known-word motion rate from accepted predictions and their captured segment durations. The underlying DTW decoding and statistical evaluation code is retained.

This is a source snapshot, not a Git submodule. Review the upstream README for its app workflow and this repository's `SUBVOX_EXPERIMENT.md` for the combined experiment.
