import Foundation

/// Collect > Export: picks one speaker's sessions, describes them in manifest.json, and zips them.
enum SessionExport {
    struct Entry: Codable, Equatable, Sendable {
        var session: String
        var speaker: String?
        var durationSeconds: Double?
        var hasAudio: Bool
        var hasWords: Bool
        /// The app quit before writing meta.json (or wrote a recovered one), so the speaker is unknown.
        var crashed: Bool

        enum CodingKeys: String, CodingKey {
            case session, speaker, crashed
            case durationSeconds = "duration_s", hasAudio = "has_audio", hasWords = "has_words"
        }
    }

    /// Sessions under `root` whose meta.json speaker matches `speaker` (case-insensitive), plus crashed ones.
    static func entries(in root: URL, speaker: String) -> [Entry] {
        let fm = FileManager.default
        let folders = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        let name = speaker.trimmingCharacters(in: .whitespaces)
        return folders.sorted { $0.lastPathComponent < $1.lastPathComponent }.compactMap { dir -> Entry? in
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return nil }
            let meta = (try? Data(contentsOf: dir.appending(path: "meta.json")))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            let crashed = meta == nil || meta?["recovered"] as? Bool == true
            let owner = (meta?["speaker"] as? String)?.trimmingCharacters(in: .whitespaces)
            guard crashed || owner?.caseInsensitiveCompare(name) == .orderedSame else { return nil }
            let has = { (file: String) in fm.fileExists(atPath: dir.appending(path: file).path) }
            return Entry(session: dir.lastPathComponent, speaker: crashed ? nil : owner,
                         durationSeconds: meta?["durationSeconds"] as? Double,
                         hasAudio: has("audio.caf") || has("audio.wav"), hasWords: has("words.json"), crashed: crashed)
        }
    }

    static func manifest(speaker: String, entries: [Entry]) throws -> Data {
        struct Manifest: Encodable { var speaker: String; var sessions: [Entry] }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(Manifest(speaker: speaker, sessions: entries))
    }

    /// Writes `zip` with manifest.json at the root and the chosen sessions under Sessions/. Blocking: call off
    /// the main actor.
    static func export(_ entries: [Entry], speaker: String, from root: URL, to zip: URL) throws {
        let fm = FileManager.default
        // Same volume as the sessions, so hard links work and nothing is copied.
        let staging = try fm.url(for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: root, create: true)
        defer { try? fm.removeItem(at: staging) }
        let sessions = staging.appending(path: "Sessions", directoryHint: .isDirectory)
        try fm.createDirectory(at: sessions, withIntermediateDirectories: true)
        for entry in entries {
            try fm.linkItem(at: root.appending(path: entry.session), to: sessions.appending(path: entry.session))
        }
        try manifest(speaker: speaker, entries: entries).write(to: staging.appending(path: "manifest.json"))

        try? fm.removeItem(at: zip)
        let ditto = Process()
        ditto.executableURL = URL(filePath: "/usr/bin/ditto")
        ditto.arguments = ["-c", "-k", staging.path, zip.path]
        let stderr = Pipe()
        ditto.standardError = stderr
        try ditto.run()
        ditto.waitUntilExit()
        guard ditto.terminationStatus == 0 else {
            let message = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            throw CocoaError(.fileWriteUnknown, userInfo: [NSLocalizedDescriptionKey: "ditto failed: \(message)"])
        }
    }
}
