import XCTest
@testable import Subvox

final class CollectPairingTests: XCTestCase {
    private var root: URL!
    private let prompt = "The birch canoe slid on the smooth planks."

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "CollectPairingTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testPairedSentenceIsSpokenThenSilentWithOnePairId() throws {
        let spoken = CollectTake.first(paired: true, id: "pair-1")
        let silent = try XCTUnwrap(spoken.next)
        XCTAssertEqual(spoken, CollectTake(condition: .normal, pairId: "pair-1"))
        XCTAssertEqual(silent, CollectTake(condition: .silent, pairId: "pair-1"))
        XCTAssertNil(silent.next, "the silent take ends the sentence")
    }

    func testPauseDropsOnlyTheSilentTake() throws {
        let spoken = CollectTake.first(paired: true)
        XCTAssertFalse(spoken.discardedOnPause, "a paused spoken take is still labelled by its own audio")
        XCTAssertTrue(try XCTUnwrap(spoken.next).discardedOnPause, "a paused silent take is likely cut short")
        XCTAssertFalse(CollectTake.first(paired: false).discardedOnPause)
    }

    func testEachSentenceGetsANewPairId() throws {
        let a = try XCTUnwrap(CollectTake.first(paired: true).pairId)
        let b = try XCTUnwrap(CollectTake.first(paired: true).pairId)
        XCTAssertNotEqual(a, b)
    }

    func testUnpairedIsOneNormalTakePerSentence() {
        let take = CollectTake.first(paired: false)
        XCTAssertEqual(take, CollectTake(condition: .normal, pairId: nil))
        XCTAssertNil(take.next)
    }

    /// Records a sentence's takes the way CollectView does and reads each meta.json back.
    @MainActor
    private func writeTakes(paired: Bool) throws -> [[String: Any]] {
        let recorder = SessionRecorder()
        recorder.prompt = prompt
        recorder.speaker = "Maya"
        var take: CollectTake? = .first(paired: paired)
        var metas: [[String: Any]] = []
        while let current = take {
            recorder.condition = current.condition
            recorder.pairId = current.pairId
            let dir = root.appending(path: "take\(metas.count)-\(current.condition.slug)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            XCTAssertTrue(SessionRecorder.writeMeta(recorder.metaFields(audio: nil), to: dir))
            let data = try Data(contentsOf: dir.appending(path: "meta.json"))
            metas.append(try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]))
            take = current.next
        }
        return metas
    }

    @MainActor
    func testPairedMetaRoundTrips() throws {
        let metas = try writeTakes(paired: true)
        XCTAssertEqual(metas.map { $0["condition"] as? String }, ["Normal speech", "Silent mouthing"])
        XCTAssertEqual(metas.map { $0["prompt"] as? String }, [prompt, prompt])
        XCTAssertEqual(metas.map { $0["speaker"] as? String }, ["Maya", "Maya"])
        let ids = metas.map { $0["pairId"] as? String }
        XCTAssertNotNil(ids[0])
        XCTAssertEqual(ids[0], ids[1])
        XCTAssertNotNil(UUID(uuidString: ids[0] ?? ""))
    }

    @MainActor
    func testUnpairedMetaHasNoPairId() throws {
        let metas = try writeTakes(paired: false)
        XCTAssertEqual(metas.count, 1)
        XCTAssertEqual(metas[0]["condition"] as? String, "Normal speech")
        XCTAssertNil(metas[0]["pairId"])
    }
}
