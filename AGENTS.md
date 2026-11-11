# Notes for coding agents

## Build and test (no Apple account needed)

```sh
brew install xcodegen
xcodegen generate
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build \
  DEVELOPMENT_TEAM= CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual test
```

Every PR must pass this. Add a unit test in `SubvoxTests/` for any logic you change.

## Rules

- Pick issues labeled `agent-ready`. Skip `needs-owner`. For `needs-hardware`, do the code and tests, and say in the PR what a human must check with AirPods Pro.
- One issue per PR. Reference it with `Fixes #N`.
- Don't commit `Subvox.xcodeproj` (generated), `build/`, `dist/`, or any recorded session data.
- Don't change the app's look. UI changes only where the issue asks.
- No new dependencies (Swift packages or pip) unless the issue says so.
- Real data lives outside the repo (`~/Documents/Subvox/Sessions`, `~/Library/Application Support/Subvox/dataset.json`). Use synthetic fixtures in tests.

## Map

- `Subvox/Motion`: AirPods motion stream (`CMHeadphoneMotionManager`), about 50 Hz.
- `Subvox/Decoding`: segmentation, features, DTW, classifier, evaluation.
- `Subvox/Recording`: session recorder and collect prompts.
- `Subvox/Views`: SwiftUI screens (Sensors, Train, Decode, Collect).
- `analysis/`: offline scoring scripts. `tools/`: labeling and DMG build.
