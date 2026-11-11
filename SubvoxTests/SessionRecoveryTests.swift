import XCTest
@testable import Subvox

final class SessionRecoveryTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "SessionRecoveryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func session(_ name: String, motionRows: Int, meta: Bool = false) throws -> URL {
        let dir = root.appending(path: name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let rows = (0..<motionRows).map { i in
            ([Double(i) * 0.02] + Array(repeating: 0, count: 16)).map { String(format: "%.6f", $0) }.joined(separator: ",")
        }
        try ([MotionSample.csvHeader] + rows).joined(separator: "\n").appending("\n")
            .write(to: dir.appending(path: "motion.csv"), atomically: true, encoding: .utf8)
        if meta { try Data("{\"condition\":\"Whisper\"}".utf8).write(to: dir.appending(path: "meta.json")) }
        return dir
    }

    private func meta(_ dir: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: dir.appending(path: "meta.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testWritesRecoveredMetaForCrashedSession() throws {
        let crashed = try session("2026-11-11T045626-123-normal", motionRows: 1951)

        XCTAssertEqual(SessionRecorder.recoverCrashedSessions(in: root).map(\.lastPathComponent), [crashed.lastPathComponent])

        let m = try meta(crashed)
        XCTAssertEqual(m["condition"] as? String, "Normal speech")
        XCTAssertEqual(m["motionSamples"] as? Int, 1951)
        XCTAssertEqual(try XCTUnwrap(m["durationSeconds"] as? Double), 39.0, accuracy: 1e-9)
        XCTAssertEqual(m["recovered"] as? Bool, true)
        XCTAssertTrue(m["audioStartOffsetSeconds"] is NSNull)
    }

    func testLeavesFinishedAndForeignFoldersAlone() throws {
        let finished = try session("2026-11-11T050000-000-whisper", motionRows: 10, meta: true)
        let foreign = try session("2026-11-11T050100-dictation", motionRows: 10)

        XCTAssertEqual(SessionRecorder.recoverCrashedSessions(in: root), [])

        XCTAssertEqual(try meta(finished) as NSDictionary, ["condition": "Whisper"] as NSDictionary)
        XCTAssertFalse(FileManager.default.fileExists(atPath: foreign.appending(path: "meta.json").path))
    }
}
