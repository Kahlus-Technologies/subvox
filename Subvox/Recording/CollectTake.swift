import Foundation

/// One Collect recording. With pairing on, each sentence is read aloud (normal), then mouthed with no
/// sound (silent): two sessions whose meta.json share a `pairId`. The silent take's audio has no words,
/// so tools/build_dataset.py gives it the spoken take's word timings by aligning the two motion streams.
struct CollectTake: Equatable {
    var condition: SessionRecorder.Condition
    var pairId: String?

    /// A sentence's first take: spoken, with a new pairId when pairing is on.
    static func first(paired: Bool, id: String = UUID().uuidString) -> CollectTake {
        CollectTake(condition: .normal, pairId: paired ? id : nil)
    }

    /// The take after this one, or nil when the sentence is done.
    var next: CollectTake? {
        guard let pairId, condition == .normal else { return nil }
        return CollectTake(condition: .silent, pairId: pairId)
    }
}
