import SwiftUI

/// Guided data collection for friends: read the sentence on screen out loud, press Next.
/// Each sentence is recorded as its own session (motion + audio), labels are added later.
struct CollectView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("collectIndex") private var index = 0
    @AppStorage("collectName") private var name = ""
    @State private var done = 0

    private var sentence: String { Prompts.harvard[index % Prompts.harvard.count] }

    var body: some View {
        let recorder = model.recorder
        let airpods = model.motion.status == .streaming
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("1. Setup") {
                    VStack(alignment: .leading, spacing: 8) {
                        check(airpods, airpods ? "AirPods motion is streaming" : "Put in your AirPods Pro and play them as this Mac's output")
                        check(model.audio.isRunning, model.audio.isRunning
                              ? "Mic on: \(model.audio.activeDeviceName ?? "default")" : "Mic starts when you press Start")
                        TextField("Your first name", text: $name).textFieldStyle(.roundedBorder).frame(maxWidth: 240)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }

                GroupBox("2. Read the sentence out loud at normal speed, keep your head still") {
                    VStack(alignment: .leading, spacing: 14) {
                        Text(sentence)
                            .font(.system(size: 30, weight: .medium))
                            .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
                            .opacity(recorder.isRecording ? 1 : 0.45)
                        ProgressView(value: Double(model.audio.level)).frame(maxWidth: 300)
                        HStack(spacing: 12) {
                            if recorder.isRecording {
                                Button("Next sentence  →") { next() }
                                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: [])
                                Button("Redo") { redo() }
                                Button("Pause") { recorder.stop() }
                            } else {
                                Button("Start") { begin() }
                                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: [])
                                    .disabled(!airpods || name.trimmingCharacters(in: .whitespaces).isEmpty)
                            }
                            Spacer()
                            Text("\(done) this sitting · sentence \(index % Prompts.harvard.count + 1) of \(Prompts.harvard.count)")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                        Text("Press Return for the next sentence. Aim for 100 sentences (about 10 minutes). Take the AirPods out and back in between sittings, that helps.")
                            .font(.caption).foregroundStyle(.secondary)
                        if let error = recorder.error { Text(error).font(.caption).foregroundStyle(.red) }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading).padding(6)
                }

                GroupBox("3. Send it back") {
                    HStack {
                        Text("Zip your recordings and send the file to Aayu.").foregroundStyle(.secondary)
                        Spacer()
                        Button("Export recordings…") { export() }.disabled(recorder.isRecording)
                    }
                    .padding(6)
                }
            }
            .padding(20)
        }
    }

    private func check(_ ok: Bool, _ text: String) -> some View {
        Label(text, systemImage: ok ? "checkmark.circle.fill" : "circle")
            .foregroundStyle(ok ? .green : .secondary)
    }

    private func begin() {
        let r = model.recorder
        r.condition = .normal
        r.prompt = sentence
        r.speaker = name.trimmingCharacters(in: .whitespaces)
        r.start(audio: model.audio)
    }

    private func next() {
        model.recorder.stop()
        index += 1
        done += 1
        begin()
    }

    private func redo() {
        let r = model.recorder
        r.stop(discard: true)
        begin()
    }

    private func export() {
        let stamp = Date().formatted(.iso8601.year().month().day())
        let safe = name.isEmpty ? "subvox" : name.lowercased().filter(\.isLetter)
        let out = URL.desktopDirectory.appending(path: "subvox-\(safe)-\(stamp).zip")
        try? FileManager.default.removeItem(at: out)
        let p = Process()
        p.executableURL = URL(filePath: "/usr/bin/ditto")
        p.arguments = ["-c", "-k", "--keepParent", SessionRecorder.root.path, out.path]
        try? p.run()
        p.waitUntilExit()
        NSWorkspace.shared.activateFileViewerSelecting([out])
    }
}
