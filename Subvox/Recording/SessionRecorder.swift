import AVFoundation
import Observation

/// Records synchronized motion CSV + microphone audio + labelled segment events for offline
/// analysis (normal speech vs. whisper vs. silent mouthing).
///
/// Output: ~/Documents/Subvox/Sessions/<timestamp>-<condition>/
///   motion.csv  — every headphone-motion sample, t relative to session start
///   audio.caf   — microphone audio (if the mic is running)
///   events.csv  — captured segments with their label / prediction
///   meta.json   — condition, prompt, clock offsets
@MainActor @Observable
final class SessionRecorder {
    enum Condition: String, CaseIterable, Identifiable, Codable {
        case normal = "Normal speech"
        case whisper = "Whisper"
        case silent = "Silent mouthing"
        var id: String { rawValue }
        var slug: String {
            switch self {
            case .normal: "normal"
            case .whisper: "whisper"
            case .silent: "silent"
            }
        }
    }

    var condition: Condition = .silent
    var prompt = ""
    var speaker = ""
    private(set) var isRecording = false
    private(set) var startedAt: Date?
    private(set) var motionCount = 0
    private(set) var eventCount = 0
    private(set) var lastSessionURL: URL?
    private(set) var error: String?

    static var root: URL {
        URL.documentsDirectory.appending(path: "Subvox/Sessions", directoryHint: .isDirectory)
    }

    @ObservationIgnored private var directory: URL?
    @ObservationIgnored private var startUptime: TimeInterval = 0
    @ObservationIgnored private var motionHandle: FileHandle?
    @ObservationIgnored private var eventsHandle: FileHandle?
    @ObservationIgnored private var audioWriter: AudioFileWriter?
    @ObservationIgnored private var sinkID: UUID?
    @ObservationIgnored private weak var sink: AudioSink?
    @ObservationIgnored private var audioDeviceName: String?

    func start(audio: AudioService) {
        guard !isRecording else { return }
        // Talk-mode labels need audio: if the mic is off, turn it on first, then start recording.
        if !audio.isRunning {
            Task {
                await audio.start()
                if audio.isRunning { start(audio: audio) } else { error = "Mic didn't start: \(audio.error ?? "unknown")" }
            }
            return
        }
        error = nil
        // Fractional seconds: the Collect prompter can start a new session within the same second.
        let stamp = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: true))
            .replacingOccurrences(of: ":", with: "").replacingOccurrences(of: ".", with: "-")
        let dir = Self.root.appending(path: "\(stamp)-\(condition.slug)", directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            motionHandle = try Self.makeFile(dir.appending(path: "motion.csv"), header: MotionSample.csvHeader)
            eventsHandle = try Self.makeFile(dir.appending(path: "events.csv"), header: "start,end,kind,label,confidence")
        } catch {
            self.error = error.localizedDescription
            return
        }

        directory = dir
        startUptime = ProcessInfo.processInfo.systemUptime
        startedAt = Date()
        motionCount = 0
        eventCount = 0

        if audio.isRunning {
            let writer = AudioFileWriter(url: dir.appending(path: "audio.caf"))
            audioWriter = writer
            sink = audio.sink
            sinkID = audio.sink.add { buffer, time in writer.write(buffer, time) }
            audioDeviceName = audio.activeDeviceName
        }
        isRecording = true
    }

    func stop(discard: Bool = false) {
        guard isRecording, let directory else { return }
        isRecording = false
        if let sinkID { sink?.remove(sinkID) }
        sinkID = nil
        let audioInfo = audioWriter?.finish()
        audioWriter = nil
        try? motionHandle?.close()
        try? eventsHandle?.close()
        motionHandle = nil
        eventsHandle = nil

        let meta: [String: Any] = [
            "condition": condition.rawValue,
            "prompt": prompt,
            "speaker": speaker,
            "startedAt": startedAt?.ISO8601Format() ?? "",
            "startUptime": startUptime,
            "durationSeconds": ProcessInfo.processInfo.systemUptime - startUptime,
            "motionSamples": motionCount,
            "events": eventCount,
            "audioDevice": audioDeviceName ?? NSNull(),
            "audioSampleRate": audioInfo?.sampleRate ?? NSNull(),
            "audioStartOffsetSeconds": audioInfo?.firstHostSeconds.map { $0 - startUptime } ?? NSNull(),
            "note": "Motion t and audioStartOffsetSeconds are relative to startUptime (seconds since boot).",
        ]
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: directory.appending(path: "meta.json"))
        }
        lastSessionURL = directory
        self.directory = nil
        if discard {
            try? FileManager.default.removeItem(at: directory)
            lastSessionURL = nil
            return
        }
        if audioInfo != nil { Self.transcribe(directory, audioOffset: meta["audioStartOffsetSeconds"] as? Double ?? 0) }
    }

    /// Talk mode labels: runs local Phonon-2 (tools/.venv, see tools/README.md) on the session audio and
    /// writes words.json with every word's start/end shifted onto the motion.csv clock.
    nonisolated private static func transcribe(_ dir: URL, audioOffset: Double) {
        let phonon = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "tools/.venv/bin/phonon")
        guard FileManager.default.isExecutableFile(atPath: phonon.path) else { return }
        DispatchQueue.global(qos: .utility).async {
            let process = Process()
            process.executableURL = phonon
            process.arguments = ["transcribe", "--json", dir.appending(path: "audio.caf").path]
            let out = Pipe()
            process.standardOutput = out
            process.standardError = FileHandle.nullDevice
            guard (try? process.run()) != nil else { return }
            let data = out.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard var json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let words = json["words"] as? [[String: Any]] else { return }
            json["words"] = words.map { word in
                var w = word
                w["start"] = (word["start"] as? Double ?? 0) + audioOffset
                w["end"] = (word["end"] as? Double ?? 0) + audioOffset
                return w
            }
            json["clock"] = "start/end are on the motion.csv t clock"
            if let outData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted]) {
                try? outData.write(to: dir.appending(path: "words.json"))
            }
        }
    }

    func record(_ sample: MotionSample) {
        guard isRecording else { return }
        motionHandle?.write(Data((sample.csvRow(relativeTo: startUptime) + "\n").utf8))
        motionCount += 1
    }

    func logEvent(_ segment: [MotionSample], kind: String, label: String, confidence: Double?) {
        guard isRecording, let first = segment.first, let last = segment.last else { return }
        let row = [
            String(format: "%.4f", first.t - startUptime),
            String(format: "%.4f", last.t - startUptime),
            kind,
            label.replacingOccurrences(of: ",", with: " "),
            confidence.map { String(format: "%.3f", $0) } ?? "",
        ].joined(separator: ",")
        eventsHandle?.write(Data((row + "\n").utf8))
        eventCount += 1
    }

    /// A crash or force quit skips `stop()`, so the session folder has motion.csv but no meta.json and
    /// tools/label_sessions.py can't tell what it is. Writes a recovered meta.json for each such folder
    /// that Subvox recorded (name ends in a condition slug). Call at launch, before any recording starts.
    @discardableResult
    nonisolated static func recoverCrashedSessions(in root: URL) -> [URL] {
        let fm = FileManager.default
        guard let folders = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        var recovered: [URL] = []
        for dir in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let motion = dir.appending(path: "motion.csv"), metaURL = dir.appending(path: "meta.json")
            guard fm.fileExists(atPath: motion.path), !fm.fileExists(atPath: metaURL.path),
                  let slug = dir.lastPathComponent.split(separator: "-").last,
                  let condition = Condition.allCases.first(where: { $0.slug == slug }),
                  let csv = try? String(contentsOf: motion, encoding: .utf8) else { continue }
            let rows = csv.split(whereSeparator: \.isNewline).dropFirst() // header
            let lastT = rows.last.flatMap { $0.split(separator: ",").first }.flatMap { Double($0) } ?? 0
            let meta: [String: Any] = [
                "condition": condition.rawValue,
                "durationSeconds": lastT,
                "motionSamples": rows.count,
                "recovered": true,
                "audioStartOffsetSeconds": NSNull(),
                "note": "Recovered at launch: the app quit before writing meta.json, so the audio offset is unknown.",
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]),
                  (try? data.write(to: metaURL)) != nil else { continue }
            recovered.append(dir)
        }
        return recovered
    }

    private static func makeFile(_ url: URL, header: String) throws -> FileHandle {
        try Data((header + "\n").utf8).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        return handle
    }
}

/// Writes tap buffers to disk from the audio thread.
private final class AudioFileWriter: @unchecked Sendable {
    struct Info {
        var sampleRate: Double
        var firstHostSeconds: Double?
    }

    private let url: URL
    private let lock = NSLock()
    private var file: AVAudioFile?
    private var info: Info?
    private var failed = false

    init(url: URL) {
        self.url = url
    }

    func write(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
        lock.withLock {
            guard !failed else { return }
            if file == nil {
                do {
                    let format = buffer.format
                    file = try AVAudioFile(
                        forWriting: url, settings: format.settings,
                        commonFormat: format.commonFormat, interleaved: format.isInterleaved)
                    info = Info(
                        sampleRate: format.sampleRate,
                        firstHostSeconds: time.isHostTimeValid ? AVAudioTime.seconds(forHostTime: time.hostTime) : nil)
                } catch {
                    failed = true
                    return
                }
            }
            try? file?.write(from: buffer)
        }
    }

    func finish() -> Info? {
        lock.withLock {
            file = nil
            return info
        }
    }
}
