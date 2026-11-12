import XCTest
@testable import Subvox

/// Faster matching (issue #9) must give the same answers bit for bit. These tests compare the app's
/// `DTW` and `TemplateClassifier.rank` with `Reference`, a frozen copy of the code they replaced.
final class RankSpeedTests: XCTestCase {
    func testDistanceMatchesReferenceExactly() {
        var rng = SplitMix64(seed: 9)
        // Lengths from 1 up, very unequal pairs, and band fractions from none to the whole grid.
        for _ in 0..<400 {
            let width = Int.random(in: 1...9, using: &rng)
            let a = randomFrames(count: Int.random(in: 1...50, using: &rng), width: width, using: &rng)
            let b = randomFrames(count: Int.random(in: 1...50, using: &rng), width: width, using: &rng)
            let fraction = [0, 0.1, 0.25, 0.5, 1.0][Int.random(in: 0..<5, using: &rng)]
            XCTAssertEqual(DTW.distance(a, b, bandFraction: fraction), Reference.distance(a, b, bandFraction: fraction))
        }
        // Different widths compare the shared leading channels; no channels at all costs nothing.
        for (widthA, widthB) in [(3, 2), (2, 5), (0, 4), (0, 0)] {
            let a = randomFrames(count: 12, width: widthA, using: &rng)
            let b = randomFrames(count: 17, width: widthB, using: &rng)
            XCTAssertEqual(DTW.distance(a, b), Reference.distance(a, b))
        }
        XCTAssertEqual(DTW.distance([], [[1, 2]]), .infinity)
        XCTAssertEqual(DTW.distance([[1, 2]], []), .infinity)
    }

    func testRankMatchesReferenceExactly() throws {
        var rng = SplitMix64(seed: 42)
        let templates = syntheticTemplates(words: 6, perWord: 12, using: &rng)
        let featureSets: [FeatureSet] = [.all, .rotation, [.acceleration, .attitude]]
        for featureSet in featureSets {
            let classifier = try XCTUnwrap(TemplateClassifier(templates: templates, featureSet: featureSet))
            let reference = Reference.Classifier(templates: templates, featureSet: featureSet)
            for _ in 0..<15 {
                let query = syntheticTake(word: Int.random(in: 0..<6, using: &rng), using: &rng)
                XCTAssertEqual(classifier.rank(query), reference.rank(query))
            }
        }
    }

    func testRankBreaksTiesByLabel() throws {
        var rng = SplitMix64(seed: 7)
        let classifier = try XCTUnwrap(TemplateClassifier(
            templates: syntheticTemplates(words: 5, perWord: 2, using: &rng), featureSet: .all))
        // An empty segment is infinitely far from every word.
        let ranked = classifier.rank([])
        XCTAssertEqual(ranked.map(\.label), ["word0", "word1", "word2", "word3", "word4"])
        XCTAssertTrue(ranked.allSatisfy { $0.distance == .infinity })
    }

    func testPairwiseDistancesMatchSerialLoop() {
        var rng = SplitMix64(seed: 3)
        let seqs = (0..<20).map { _ in
            randomFrames(count: Int.random(in: 5...40, using: &rng), width: 9, using: &rng)
        }
        let matrix = Evaluation.pairwiseDistances(seqs.map(DTW.Frames.init))
        for i in seqs.indices {
            for j in seqs.indices {
                // The old loop computed each pair once as (lower index, higher index) and mirrored it.
                let expected = i == j ? 0 : Reference.distance(seqs[min(i, j)], seqs[max(i, j)])
                XCTAssertEqual(matrix[i][j], expected)
            }
        }
    }

    // MARK: Benchmarks. 1,000 templates of 10 words, one query per iteration.

    func testRankSpeed() throws {
        let (templates, query) = benchmarkData()
        let classifier = try XCTUnwrap(TemplateClassifier(templates: templates, featureSet: .all))
        measure { _ = classifier.rank(query) }
    }

    func testReferenceRankSpeed() {
        let (templates, query) = benchmarkData()
        let reference = Reference.Classifier(templates: templates, featureSet: .all)
        measure { _ = reference.rank(query) }
    }

    private func benchmarkData() -> ([Template], [MotionSample]) {
        var rng = SplitMix64(seed: 1_000)
        return (syntheticTemplates(words: 10, perWord: 100, using: &rng), syntheticTake(word: 3, using: &rng))
    }
}

// MARK: Synthetic data

/// Deterministic generator, so a failure reproduces.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func randomFrames(count: Int, width: Int, using rng: inout SplitMix64) -> [[Double]] {
    (0..<count).map { _ in (0..<width).map { _ in Double.random(in: -2...2, using: &rng) } }
}

/// A word is a burst on one rotation and one acceleration axis at a word-specific rhythm,
/// with noise on every channel. Takes run 20 to 60 samples, 0.4 to 1.2 s at 50 Hz.
private func syntheticTake(word: Int, using rng: inout SplitMix64) -> [MotionSample] {
    let count = Int.random(in: 20...60, using: &rng)
    let frequency = 1 + 0.7 * Double(word % 4)
    var samples: [MotionSample] = []
    for i in 0..<count {
        let phase = Double(i) / Double(count - 1)
        let v = 0.2 * sin(.pi * phase) * sin(2 * .pi * frequency * phase)
        var channels = (0..<9).map { _ in Double.random(in: -0.02...0.02, using: &rng) }
        channels[3 + word % 3] += v
        channels[(word / 3) % 3] += 0.5 * v
        channels[6 + word % 3] += 0.3 * v * phase
        samples.append(MotionSample(
            t: Double(i) / 50,
            accel: (channels[0], channels[1], channels[2]),
            rotation: (channels[3], channels[4], channels[5]),
            attitude: (channels[6], channels[7], channels[8])))
    }
    return samples
}

private func syntheticTemplates(words: Int, perWord: Int, using rng: inout SplitMix64) -> [Template] {
    var templates: [Template] = []
    for _ in 0..<perWord {
        for word in 0..<words {
            templates.append(Template(label: "word\(word)", samples: syntheticTake(word: word, using: &rng)))
        }
    }
    return templates
}

// MARK: Reference

/// `DTW` and `TemplateClassifier.rank` as they were before issue #9. Do not optimise this copy.
private enum Reference {
    struct Classifier {
        let featureSet: FeatureSet
        let entries: [(label: String, frames: [[Double]])]
        let scales: [Double]

        init(templates: [Template], featureSet: FeatureSet) {
            let raw = templates.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
            let scales = MotionFeatures.channelScales(raw)
            self.featureSet = featureSet
            self.scales = scales
            self.entries = zip(templates, raw).map { ($0.label, MotionFeatures.scaled($1, by: scales)) }
        }

        func rank(_ samples: [MotionSample]) -> [RankedLabel] {
            let query = MotionFeatures.scaled(
                MotionFeatures.frames(MotionFeatures.trim(samples), set: featureSet), by: scales)
            var best: [String: Double] = [:]
            for entry in entries {
                let d = Reference.distance(query, entry.frames)
                best[entry.label] = min(best[entry.label] ?? .infinity, d)
            }
            return best.map { RankedLabel(label: $0.key, distance: $0.value) }.sorted { $0.distance < $1.distance }
        }
    }

    static func distance(_ a: [[Double]], _ b: [[Double]], bandFraction: Double = 0.25) -> Double {
        let n = a.count, m = b.count
        guard n > 0, m > 0 else { return .infinity }
        let band = max(abs(n - m), Int(Double(max(n, m)) * bandFraction))

        var previous = [Double](repeating: .infinity, count: m + 1)
        var current = previous
        previous[0] = 0

        for i in 1...n {
            for j in current.indices { current[j] = .infinity }
            let lo = max(1, i - band)
            let hi = min(m, i + band)
            if lo <= hi {
                for j in lo...hi {
                    let cost = euclidean(a[i - 1], b[j - 1])
                    current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
                }
            }
            swap(&previous, &current)
        }
        return previous[m] / Double(n + m)
    }

    private static func euclidean(_ x: [Double], _ y: [Double]) -> Double {
        var sum = 0.0
        for k in 0..<min(x.count, y.count) {
            let d = x[k] - y[k]
            sum += d * d
        }
        return sum.squareRoot()
    }
}
