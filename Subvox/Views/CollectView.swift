import SwiftUI

/// Guided data collection for friends: read the sentence on screen out loud, press Next.
/// Each take is recorded as its own session (motion + audio), labels are added later. With pairing on,
/// each sentence is read aloud, then mouthed silently (see CollectTake).
struct CollectView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("collectIndex") private var index = 0
    @AppStorage("collectName") private var name = ""
    @AppStorage("collectPaired") private var paired = true
    @State private var take = CollectTake.first(paired: true)
    @State private var done = 0
    @State private var mine: [SessionExport.Entry] = []
    @State private var isExporting = false
    @State private var exportProblem: String?

    private var sentence: String { Prompts.harvard[index % Prompts.harvard.count] }

    private var progress: String {
        let finished = paired ? "\(done) \(done == 1 ? "pair" : "pairs")" : "\(done)"
        return "\(finished) this sitting · sentence \(index % Prompts.harvard.count + 1) of \(Prompts.harvard.count)"
    }

    var body: some View {
        let recorder = model.recorder
        let airpods = model.motion.status == .streaming
        let mouthing = recorder.isRecording && take.condition == .silent
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("1. Setup") {
                    VStack(alignment: .leading, spacing: 8) {
                        check(airpods, airpods ? "AirPods motion is streaming" : "Put in your AirPods Pro and play them as this Mac's output")
                        check(model.audio.isRunning, model.audio.isRunning
                              ? "Mic on: \(model.audio.activeDeviceName ?? "default")" : "Mic starts when you press Start")
                        TextField("Your first name", text: $name).textFieldStyle(.roundedBorder).frame(maxWidth: 240)
                        Toggle("After reading each sentence aloud, mouth it again with no sound", isOn: $paired)
                            .disabled(recorder.isRecording)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }

                GroupBox(mouthing ? "2. Now mouth the same sentence with no sound, keep your head still"
                                  : "2. Read the sentence out loud at normal speed, keep your head still") {
                    VStack(alignment: .leading, spacing: 14) {
                        Text(sentence)
                            .font(.system(size: 30, weight: .medium))
                            .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
                            .opacity(recorder.isRecording ? 1 : 0.45)
                        ProgressView(value: Double(model.audio.level)).frame(maxWidth: 300)
                        HStack(spacing: 12) {
                            if recorder.isRecording {
                                Button(take.next == nil ? "Next sentence  →" : "Now mouth it  →") { next() }
                                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: [])
                                Button("Redo") { redo() }
                                Button("Pause") { recorder.stop() }
                            } else {
                                Button("Start") { begin() }
                                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: [])
                                    .disabled(!airpods || name.trimmingCharacters(in: .whitespaces).isEmpty)
                            }
                            Spacer()
                            Text(progress)
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                        Text(paired
                             ? "Press Return after each take. Aim for 100 sentences (about 20 minutes). Take the AirPods out and back in between sittings, that helps."
                             : "Press Return for the next sentence. Aim for 100 sentences (about 10 minutes). Take the AirPods out and back in between sittings, that helps.")
                            .font(.caption).foregroundStyle(.secondary)
                        if let error = recorder.error { Text(error).font(.caption).foregroundStyle(.red) }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }

                GroupBox("3. Send it back") {
                    HStack {
                        Text("Zip your recordings and send the file to Aayu.").foregroundStyle(.secondary)
                        Spacer()
                        let minutes = mine.compactMap(\.durationSeconds).reduce(0, +) / 60
                        Text("\(mine.count) recordings · \(minutes, format: .number.precision(.fractionLength(1))) min")
                            .foregroundStyle(.secondary).monospacedDigit()
                        if isExporting { ProgressView().controlSize(.small) }
                        Button("Export recordings…") { export() }.disabled(recorder.isRecording || isExporting)
                    }
                    .padding(6)
                }
            }
            .padding(20)
        }
        .task(id: "\(name)|\(recorder.isRecording)") {
            let speaker = name, root = SessionRecorder.root
            mine = await Task.detached { SessionExport.entries(in: root, speaker: speaker) }.value
        }
        .alert("Couldn't export", isPresented: Binding(get: { exportProblem != nil }, set: { if !$0 { exportProblem = nil } })) {
        } message: {
            Text(exportProblem ?? "")
        }
    }

    private func check(_ ok: Bool, _ text: String) -> some View {
        Label(text, systemImage: ok ? "checkmark.circle.fill" : "circle")
            .foregroundStyle(ok ? .green : .secondary)
    }

    /// Start (or resume after Pause): a new pair from the spoken take of the current sentence.
    private func begin() {
        take = .first(paired: paired)
        record()
    }

    private func record() {
        let r = model.recorder
        r.condition = take.condition
        r.pairId = take.pairId
        r.prompt = sentence
        r.speaker = name.trimmingCharacters(in: .whitespaces)
        r.start(audio: model.audio)
    }

    private func next() {
        model.recorder.stop()
        if let following = take.next {
            take = following
            record()
            return
        }
        index += 1
        done += 1
        begin()
    }

    private func redo() {
        model.recorder.stop(discard: true)
        record()
    }

    private func export() {
        let stamp = Date().formatted(.iso8601.year().month().day())
        let safe = name.isEmpty ? "subvox" : name.lowercased().filter(\.isLetter)
        let out = URL.desktopDirectory.appending(path: "subvox-\(safe)-\(stamp).zip")
        let speaker = name.trimmingCharacters(in: .whitespaces), root = SessionRecorder.root
        isExporting = true
        Task {
            defer { isExporting = false }
            do {
                let entries = try await Task.detached {
                    let entries = SessionExport.entries(in: root, speaker: speaker)
                    if !entries.isEmpty { try SessionExport.export(entries, speaker: speaker, from: root, to: out) }
                    return entries
                }.value
                mine = entries
                if entries.isEmpty {
                    exportProblem = "There are no recordings for \"\(speaker)\" yet. Check the name in step 1."
                } else {
                    NSWorkspace.shared.activateFileViewerSelecting([out])
                }
            } catch {
                exportProblem = error.localizedDescription
            }
        }
    }
}
