import XCTest
@testable import Subvox

final class SessionExportTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "SessionExportTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func session(_ name: String, meta: [String: Any]?, files: [String] = ["motion.csv"]) throws {
        let dir = root.appending(path: name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in files { try Data("x".utf8).write(to: dir.appending(path: file)) }
        if let meta { try JSONSerialization.data(withJSONObject: meta).write(to: dir.appending(path: "meta.json")) }
    }

    private func fixture() throws {
        try session("a-normal", meta: ["speaker": "Maya", "durationSeconds": 30.0], files: ["motion.csv", "audio.caf", "words.json"])
        try session("b-normal", meta: ["speaker": " maya ", "durationSeconds": 90.0], files: ["motion.csv", "audio.caf"])
        try session("c-normal", meta: ["speaker": "Aayu", "durationSeconds": 60.0])
        try session("d-normal", meta: nil, files: ["motion.csv", "audio.caf"])
        try session("e-normal", meta: ["recovered": true, "durationSeconds": 12.0])
        try session("f-silent", meta: ["speaker": ""])
    }

    func testPicksTheSpeakersSessionsAndCrashedOnes() throws {
        try fixture()
        let entries = SessionExport.entries(in: root, speaker: "MAYA")
        XCTAssertEqual(entries, [
            .init(session: "a-normal", speaker: "Maya", durationSeconds: 30, hasAudio: true, hasWords: true, crashed: false),
            .init(session: "b-normal", speaker: "maya", durationSeconds: 90, hasAudio: true, hasWords: false, crashed: false),
            .init(session: "d-normal", speaker: nil, durationSeconds: nil, hasAudio: true, hasWords: false, crashed: true),
            .init(session: "e-normal", speaker: nil, durationSeconds: 12, hasAudio: false, hasWords: false, crashed: true),
        ])
    }

    func testManifestFormat() throws {
        let entry = SessionExport.Entry(session: "d-normal", speaker: nil, durationSeconds: nil, hasAudio: true, hasWords: false, crashed: true)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: SessionExport.manifest(speaker: "Maya", entries: [entry])) as? [String: Any])
        XCTAssertEqual(json["speaker"] as? String, "Maya")
        let first = try XCTUnwrap((json["sessions"] as? [[String: Any]])?.first)
        XCTAssertEqual(first["session"] as? String, "d-normal")
        XCTAssertEqual(first["has_audio"] as? Bool, true)
        XCTAssertEqual(first["has_words"] as? Bool, false)
        XCTAssertEqual(first["crashed"] as? Bool, true)
    }

    func testZipHoldsManifestAndOnlyTheChosenSessions() throws {
        try fixture()
        let zip = root.appending(path: "out.zip")
        try SessionExport.export(SessionExport.entries(in: root, speaker: "maya"), speaker: "maya", from: root, to: zip)

        let list = Process(), out = Pipe()
        list.executableURL = URL(filePath: "/usr/bin/zipinfo")
        list.arguments = ["-1", zip.path]
        list.standardOutput = out
        try list.run()
        list.waitUntilExit()
        let names = Set(String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).split(separator: "\n").map(String.init))
        XCTAssertTrue(names.contains("manifest.json"))
        XCTAssertTrue(names.contains("Sessions/a-normal/words.json"))
        XCTAssertTrue(names.contains("Sessions/d-normal/audio.caf"))
        XCTAssertFalse(names.contains { $0.hasPrefix("Sessions/c-normal") || $0.hasPrefix("Sessions/f-silent") })
        // The originals are untouched.
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appending(path: "a-normal/words.json").path))
    }
}
