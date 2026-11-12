import XCTest
@testable import Subvox

final class QuickSetupTests: XCTestCase {
    func testDefaultRunIsSixCommandWordsFiveTakesEach() {
        let setup = QuickSetup()
        XCTAssertEqual(setup.words, ["yes", "no", "next", "back", "play", "stop"])
        XCTAssertEqual(setup.takesPerWord, 5)
        XCTAssertEqual(setup.totalTakes, 30)
        XCTAssertEqual(setup.step, .take(word: "yes", number: 1))
        XCTAssertTrue(setup.isRunning)
        XCTAssertEqual(setup.progress, 0)
    }

    func testEachAcceptedTakeAdvancesInWordOrder() {
        var setup = QuickSetup()
        var prompted: [QuickSetup.Step] = []
        var filed: [String] = []
        while setup.isRunning {
            prompted.append(setup.step)
            filed.append(setup.accept(UUID()) ?? "?")
        }
        let expected = QuickSetup.commands.flatMap { word in (1...5).map { QuickSetup.Step.take(word: word, number: $0) } }
        XCTAssertEqual(prompted, expected)
        XCTAssertEqual(filed, QuickSetup.commands.flatMap { Array(repeating: $0, count: 5) })
        XCTAssertEqual(setup.takes.count, 30)
    }

    func testCompletionStopsAcceptingTakes() {
        var setup = QuickSetup(words: ["yes", "no"], takesPerWord: 2)
        for _ in 0..<4 { setup.accept(UUID()) }
        XCTAssertEqual(setup.step, .done)
        XCTAssertFalse(setup.isRunning)
        XCTAssertNil(setup.currentWord)
        XCTAssertEqual(setup.progress, 1)

        XCTAssertNil(setup.accept(UUID()))
        XCTAssertEqual(setup.takes.count, 4)
    }

    func testRedoDropsLastTakeAndPromptsItAgain() {
        var setup = QuickSetup(words: ["yes", "no"], takesPerWord: 2)
        let first = UUID(), second = UUID(), third = UUID()
        setup.accept(first)
        setup.accept(second)
        setup.accept(third)
        XCTAssertEqual(setup.step, .take(word: "no", number: 2))

        XCTAssertEqual(setup.redo(), third)
        XCTAssertEqual(setup.step, .take(word: "no", number: 1))
        // Stepping back over a word boundary returns to the previous word.
        XCTAssertEqual(setup.redo(), second)
        XCTAssertEqual(setup.step, .take(word: "yes", number: 2))
        XCTAssertEqual(setup.takes, [first])

        XCTAssertEqual(setup.redo(), first)
        XCTAssertNil(setup.redo(), "nothing left to redo")
        XCTAssertEqual(setup.step, .take(word: "yes", number: 1))
    }

    func testRedoAfterTheLastTakeReopensTheRun() {
        var setup = QuickSetup(words: ["yes"], takesPerWord: 2)
        setup.accept(UUID())
        let last = UUID()
        setup.accept(last)
        XCTAssertEqual(setup.step, .done)

        XCTAssertEqual(setup.redo(), last)
        XCTAssertEqual(setup.step, .take(word: "yes", number: 2))
        XCTAssertTrue(setup.isRunning)
    }

    func testCancelReturnsThisRunsTakesAndStopsEverything() {
        var setup = QuickSetup(words: ["yes", "no"], takesPerWord: 2)
        let ids = [UUID(), UUID(), UUID()]
        ids.forEach { setup.accept($0) }

        XCTAssertEqual(setup.cancel(), ids)
        XCTAssertEqual(setup.step, .cancelled)
        XCTAssertFalse(setup.isRunning)
        XCTAssertNil(setup.currentWord)
        XCTAssertNil(setup.accept(UUID()))
        XCTAssertNil(setup.redo())
        XCTAssertEqual(setup.cancel(), [], "a second cancel has nothing to delete")
    }

    func testCancelAfterCompletionKeepsTheTakes() {
        var setup = QuickSetup(words: ["yes"], takesPerWord: 1)
        setup.accept(UUID())
        XCTAssertEqual(setup.cancel(), [])
        XCTAssertEqual(setup.step, .done)
    }
}
