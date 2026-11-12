import Foundation

struct Template: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    var label: String
    var samples: [MotionSample]
    var createdAt = Date()

    var duration: TimeInterval {
        guard let first = samples.first, let last = samples.last else { return 0 }
        return last.t - first.t
    }
}

struct RankedLabel: Identifiable, Hashable, Sendable {
    var label: String
    var distance: Double
    var id: String { label }
}

struct Prediction: Hashable, Sendable {
    var ranked: [RankedLabel]
    /// 0…1 margin between the best and second-best class.
    var confidence: Double
    var accepted: Bool

    var best: RankedLabel? { ranked.first }
}

/// Small-vocabulary recognizer: nearest-neighbour DTW against user-recorded templates.
/// Deliberately simple — the question this prototype answers is whether the signal is
/// there at all, not how far a big model can push it.
struct TemplateClassifier: Sendable {
    let featureSet: FeatureSet
    private let entries: [(label: String, frames: DTW.Frames)]
    private let scales: [Double]

    var labels: [String] { Array(Set(entries.map(\.label))).sorted() }

    init?(templates: [Template], featureSet: FeatureSet) {
        guard !featureSet.isEmpty else { return nil }
        let raw = templates.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
        guard Set(templates.map(\.label)).count >= 2 else { return nil }
        let scales = MotionFeatures.channelScales(raw)
        self.featureSet = featureSet
        self.scales = scales
        self.entries = zip(templates, raw).map { ($0.label, DTW.Frames(MotionFeatures.scaled($1, by: scales))) }
    }

    func rank(_ samples: [MotionSample]) -> [RankedLabel] {
        let query = DTW.Frames(MotionFeatures.scaled(
            MotionFeatures.frames(MotionFeatures.trim(samples), set: featureSet), by: scales))
        // The DTW calls run in parallel; the per-word minimum is then taken in template order, as before.
        let distances = [Double](concurrentlyMapping: entries.count) { DTW.distance(query, entries[$0].frames) }
        var best: [String: Double] = [:]
        for (entry, d) in zip(entries, distances) {
            best[entry.label] = min(best[entry.label] ?? .infinity, d)
        }
        return best.map { RankedLabel(label: $0.key, distance: $0.value) }.sorted { $0.distance < $1.distance }
    }

    func predict(_ samples: [MotionSample], maxDistance: Double, minConfidence: Double) -> Prediction {
        let ranked = rank(samples)
        let confidence = Self.margin(ranked.map(\.distance))
        let accepted = (ranked.first?.distance ?? .infinity) <= maxDistance && confidence >= minConfidence
        return Prediction(ranked: ranked, confidence: confidence, accepted: accepted)
    }

    static func margin(_ sortedDistances: [Double]) -> Double {
        guard sortedDistances.count >= 2, sortedDistances[1] > 0 else { return 0 }
        return max(0, 1 - sortedDistances[0] / sortedDistances[1])
    }
}

/// Leave-one-out accuracy plus a label-permutation test: the honest answer to
/// "is the motion stream statistically distinguishable between words?"
struct Evaluation: Sendable {
    var labels: [String]
    /// confusion[actual][predicted]
    var confusion: [[Int]]
    var accuracy: Double
    /// Train on the first 60% of takes by `createdAt`, test on the rest (as in analysis/score.py). Takes
    /// recorded back to back look alike, so this is the honest number; leave-one-out overstates it.
    var timeSplitAccuracy: Double
    /// Accuracy of always guessing the most common label.
    var chance: Double
    var permutationMean: Double
    var permutationP95: Double
    var pValue: Double
    var permutations: Int
    var templateCount: Int

    static func run(templates: [Template], featureSet: FeatureSet, permutations: Int = 500) -> Evaluation? {
        let labels = Array(Set(templates.map(\.label))).sorted()
        let counts = Dictionary(grouping: templates, by: \.label).mapValues(\.count)
        guard labels.count >= 2, counts.values.allSatisfy({ $0 >= 2 }), !featureSet.isEmpty else { return nil }

        let raw = templates.map { MotionFeatures.frames(MotionFeatures.trim($0.samples), set: featureSet) }
        let scales = MotionFeatures.channelScales(raw)
        let seqs = raw.map { DTW.Frames(MotionFeatures.scaled($0, by: scales)) }
        let n = seqs.count

        // Pairwise distances once; every labelling (true or permuted) reuses them.
        let distance = pairwiseDistances(seqs)

        let index = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })
        let truth = templates.map { index[$0.label]! }

        func leaveOneOut(_ assignment: [Int]) -> [Int] {
            (0..<n).map { i in
                var best = [Double](repeating: .infinity, count: labels.count)
                for j in 0..<n where j != i {
                    best[assignment[j]] = min(best[assignment[j]], distance[i][j])
                }
                return best.indices.min { best[$0] < best[$1] }!
            }
        }

        func accuracy(_ predicted: [Int], _ actual: [Int]) -> Double {
            Double(zip(predicted, actual).filter { $0 == $1 }.count) / Double(n)
        }

        let predicted = leaveOneOut(truth)
        var confusion = [[Int]](repeating: [Int](repeating: 0, count: labels.count), count: labels.count)
        for (p, a) in zip(predicted, truth) { confusion[a][p] += 1 }
        let observed = accuracy(predicted, truth)

        let byTime = templates.indices.sorted { (templates[$0].createdAt, $0) < (templates[$1].createdAt, $1) }
        let cut = Int(Double(n) * 0.6)
        let train = byTime[..<cut], test = byTime[cut...]
        let timeSplitHits = test.filter { i in truth[train.min { distance[i][$0] < distance[i][$1] }!] == truth[i] }.count

        var generator = SystemRandomNumberGenerator()
        var nullAccuracies: [Double] = []
        nullAccuracies.reserveCapacity(permutations)
        for _ in 0..<permutations {
            let shuffled = truth.shuffled(using: &generator)
            nullAccuracies.append(accuracy(leaveOneOut(shuffled), shuffled))
        }
        nullAccuracies.sort()
        let atLeast = nullAccuracies.filter { $0 >= observed }.count

        return Evaluation(
            labels: labels,
            confusion: confusion,
            accuracy: observed,
            timeSplitAccuracy: Double(timeSplitHits) / Double(test.count),
            chance: Double(counts.values.max()!) / Double(n),
            permutationMean: nullAccuracies.reduce(0, +) / Double(max(permutations, 1)),
            permutationP95: nullAccuracies.isEmpty ? 0 : nullAccuracies[Int(Double(nullAccuracies.count - 1) * 0.95)],
            pValue: Double(atLeast + 1) / Double(permutations + 1),
            permutations: permutations,
            templateCount: n)
    }

    /// Symmetric DTW distance matrix with a zero diagonal. Entry [i][j] for i < j is
    /// `DTW.distance(seqs[i], seqs[j])`, mirrored to [j][i]. Rows run in parallel.
    static func pairwiseDistances(_ seqs: [DTW.Frames]) -> [[Double]] {
        let n = seqs.count
        let upper = [[Double]](concurrentlyMapping: n) { i in
            ((i + 1)..<n).map { j in DTW.distance(seqs[i], seqs[j]) }
        }
        var distance = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for (offset, d) in upper[i].enumerated() {
                distance[i][i + 1 + offset] = d
                distance[i + 1 + offset][i] = d
            }
        }
        return distance
    }
}

private extension Array {
    /// `(0..<count).map(make)`, with the calls spread across all cores. Results keep index order.
    /// `make` runs on several threads at once, so it must only read shared state.
    init(concurrentlyMapping count: Int, _ make: (Int) -> Element) {
        self.init(unsafeUninitializedCapacity: count) { buffer, initialized in
            guard let base = buffer.baseAddress else { return }
            DispatchQueue.concurrentPerform(iterations: count) { i in
                (base + i).initialize(to: make(i))
            }
            initialized = count
        }
    }
}
