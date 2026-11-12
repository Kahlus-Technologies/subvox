import AppKit
import ApplicationServices

/// System media keys, by their NX_KEYTYPE_* codes from IOKit/hidsystem/ev_keymap.h.
enum MediaKey: Int, CaseIterable {
    case playPause = 16
    case next = 17
    case previous = 18

    /// The key a decoded command word presses. Words without an action (yes, no, anything else) return nil.
    init?(command: String) {
        switch command.lowercased() {
        case "play", "stop": self = .playPause
        case "next": self = .next
        case "back": self = .previous
        default: return nil
        }
    }

    /// `data1` of the NSEvent.systemDefined (subtype 8) event for this key: code in bits 16-31,
    /// key state in bits 8-15 (0xA down, 0xB up).
    func eventData(down: Bool) -> Int {
        (rawValue << 16) | (Self.keyState(down: down) << 8)
    }

    static func keyState(down: Bool) -> Int { down ? 0xA : 0xB }
}

protocol MediaKeyPosting {
    func post(_ key: MediaKey)
}

/// Presses a media key system-wide, the same way the keyboard's media keys do.
/// macOS drops these events unless Subvox is trusted under Privacy & Security > Accessibility.
struct SystemMediaKeyPoster: MediaKeyPosting {
    func post(_ key: MediaKey) {
        for down in [true, false] {
            let event = NSEvent.otherEvent(
                with: .systemDefined,
                location: .zero,
                modifierFlags: NSEvent.ModifierFlags(rawValue: UInt(MediaKey.keyState(down: down) << 8)),
                timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: 0,
                context: nil,
                subtype: 8,
                data1: key.eventData(down: down),
                data2: -1)
            event?.cgEvent?.post(tap: .cghidEventTap)
        }
    }
}

/// Decode's optional action: an accepted command word presses its media key. Off unless the
/// Decode toggle (stored under `defaultsKey`) is on.
struct MediaKeyHook {
    static let defaultsKey = "decodeMediaKeys"

    var defaults: UserDefaults = .standard
    var poster: any MediaKeyPosting = SystemMediaKeyPoster()

    var isEnabled: Bool { defaults.bool(forKey: Self.defaultsKey) }

    /// Presses the key for `prediction` if the hook is on and the word was accepted. Returns the key pressed.
    @discardableResult
    func handle(_ prediction: Prediction) -> MediaKey? {
        guard isEnabled, prediction.accepted, let label = prediction.best?.label,
              let key = MediaKey(command: label) else { return nil }
        poster.post(key)
        return key
    }
}

enum AccessibilityTrust {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Asks macOS to show its Accessibility prompt if Subvox isn't trusted yet. Returns whether it is trusted now.
    @discardableResult
    static func request() -> Bool {
        let prompt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([prompt: true] as CFDictionary)
    }
}
