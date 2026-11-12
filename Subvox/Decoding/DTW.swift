import Foundation

/// Dynamic time warping between two multichannel sequences, with a Sakoe–Chiba band.
/// Lets the same mouthed word match when said faster or slower.
enum DTW {
    /// Equal-width frames packed back to back in one array, so the inner loop reads contiguous
    /// memory. Pack a sequence once and reuse it across `distance` calls.
    struct Frames: Sendable {
        let count: Int
        let width: Int
        let values: [Double]

        init(_ frames: [[Double]]) {
            let width = frames.first?.count ?? 0
            precondition(frames.allSatisfy { $0.count == width }, "DTW frames must all have the same width")
            self.count = frames.count
            self.width = width
            self.values = Array(frames.joined())
        }
    }

    static func distance(_ a: [[Double]], _ b: [[Double]], bandFraction: Double = 0.25) -> Double {
        distance(Frames(a), Frames(b), bandFraction: bandFraction)
    }

    static func distance(_ a: Frames, _ b: Frames, bandFraction: Double = 0.25) -> Double {
        let n = a.count, m = b.count
        guard n > 0, m > 0 else { return .infinity }
        // band >= |n - m| keeps lo <= hi on every row and puts the end cell (n, m) inside the band.
        let band = max(abs(n - m), Int(Double(max(n, m)) * bandFraction))
        let channels = min(a.width, b.width)
        // With no shared channels every frame pair costs 0, and so does the best path.
        guard channels > 0 else { return 0 }

        return a.values.withUnsafeBufferPointer { x in
            b.values.withUnsafeBufferPointer { y in
                withUnsafeTemporaryAllocation(of: Double.self, capacity: 2 * (m + 1)) { rows in
                    // Two reused rows of the n×m cost grid. A row is only written inside the band plus
                    // one infinite cell on each side, which is every cell the next row reads.
                    var previous = rows.baseAddress!
                    var current = previous + (m + 1)
                    previous.initialize(repeating: .infinity, count: m + 1)
                    previous[0] = 0

                    // The two inner loops are `while` loops on purpose. In Debug builds a for-in over a
                    // range runs generic iterator code per element, which made this over 10x slower there.
                    // Frame pointers keep the overflow check of x[start + k] out of the channel loop. With
                    // that check, Release builds ran slower than the old array-of-arrays code.
                    for i in 1...n {
                        let lo = max(1, i - band)
                        let hi = min(m, i + band)
                        let xFrame = x.baseAddress! + (i - 1) * a.width
                        current[lo - 1] = .infinity
                        var j = lo
                        while j <= hi {
                            let yFrame = y.baseAddress! + (j - 1) * b.width
                            var sum = 0.0
                            var k = 0
                            while k < channels {
                                let d = xFrame[k] - yFrame[k]
                                sum += d * d
                                k += 1
                            }
                            current[j] = sum.squareRoot() + min(previous[j], current[j - 1], previous[j - 1])
                            j += 1
                        }
                        if hi < m { current[hi + 1] = .infinity }
                        swap(&previous, &current)
                    }
                    // Normalize by the maximum path length so long and short words compare fairly.
                    return previous[m] / Double(n + m)
                }
            }
        }
    }
}
