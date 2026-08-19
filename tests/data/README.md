# Test fixtures

`jfk.wav` (JFK "ask not what your country can do for you" excerpt, 16 kHz
16-bit mono) is used by the VAD/STT/selftest integration tests.

It is **not committed** to the repository — fetch it (and the models) with:

```bash
./scripts/download_models.sh
```

Tests skip with a printed reason when the fixture is absent.
