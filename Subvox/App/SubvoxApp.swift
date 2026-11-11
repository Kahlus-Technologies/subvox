import SwiftUI

@main
struct SubvoxApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .frame(minWidth: 980, minHeight: 680)
                .task { model.start() }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    model.recorder.stop() // quitting mid-sentence still saves meta.json
                }
        }
        .windowResizability(.contentMinSize)
    }
}
