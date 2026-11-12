import XCTest
@testable import Subvox

final class MediaKeyTests: XCTestCase {
    private final class SpyPoster: MediaKeyPosting {
        var posted: [MediaKey] = []
        func post(_ key: MediaKey) { posted.append(key) }
    }

    private var defaults: UserDefaults!
    private let suite = "MediaKeyTests"
    private var spy: SpyPoster!

    override func setUp() {
        defaults = UserDefaults(suiteName: suite)
        defaults.removePersistentDomain(forName: suite)
        spy = SpyPoster()
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    private var hook: MediaKeyHook { MediaKeyHook(defaults: defaults, poster: spy) }

    private func prediction(_ label: String, accepted: Bool = true) -> Prediction {
        Prediction(ranked: [RankedLabel(label: label, distance: 0.5), RankedLabel(label: "other", distance: 1)],
                   confidence: 0.5, accepted: accepted)
    }

    func testCommandWordsMapToMediaKeys() {
        XCTAssertEqual(MediaKey(command: "play"), .playPause)
        XCTAssertEqual(MediaKey(command: "stop"), .playPause)
        XCTAssertEqual(MediaKey(command: "next"), .next)
        XCTAssertEqual(MediaKey(command: "back"), .previous)
        XCTAssertEqual(MediaKey(command: "Next"), .next)
    }

    func testYesNoAndOtherWordsPressNothing() {
        for word in ["yes", "no", "hello", ""] {
            XCTAssertNil(MediaKey(command: word), word)
        }
    }

    func testKeyCodesAndEventData() {
        XCTAssertEqual(MediaKey.playPause.rawValue, 16) // NX_KEYTYPE_PLAY
        XCTAssertEqual(MediaKey.next.rawValue, 17) // NX_KEYTYPE_NEXT
        XCTAssertEqual(MediaKey.previous.rawValue, 18) // NX_KEYTYPE_PREVIOUS
        XCTAssertEqual(MediaKey.playPause.eventData(down: true), 0x10_0A00)
        XCTAssertEqual(MediaKey.playPause.eventData(down: false), 0x10_0B00)
        XCTAssertEqual(MediaKey.previous.eventData(down: true), (18 << 16) | (0xA << 8))
    }

    func testHookIsOffByDefaultAndDoesNothingWhileOff() {
        XCTAssertFalse(hook.isEnabled)
        for word in QuickSetup.commands {
            XCTAssertNil(hook.handle(prediction(word)))
        }
        defaults.set(false, forKey: MediaKeyHook.defaultsKey)
        XCTAssertNil(hook.handle(prediction("play")))
        XCTAssertEqual(spy.posted, [])
    }

    func testHookPressesKeysForAcceptedCommandsWhenOn() {
        defaults.set(true, forKey: MediaKeyHook.defaultsKey)
        for word in QuickSetup.commands { hook.handle(prediction(word)) }
        XCTAssertEqual(spy.posted, [.next, .previous, .playPause, .playPause])
    }

    func testHookIgnoresRejectedPredictions() {
        defaults.set(true, forKey: MediaKeyHook.defaultsKey)
        XCTAssertNil(hook.handle(prediction("play", accepted: false)))
        XCTAssertNil(hook.handle(Prediction(ranked: [], confidence: 0, accepted: true)))
        XCTAssertEqual(spy.posted, [])
    }
}
