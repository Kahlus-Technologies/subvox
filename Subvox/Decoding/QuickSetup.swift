import Foundation

/// Progress through Train's Quick setup: each command word in order, a fixed number of takes per word.
/// It only counts takes. AppModel stores the templates and runs capture; the view shows `step`.
struct QuickSetup: Equatable {
    static let commands = ["yes", "no", "next", "back", "play", "stop"]
    static let defaultTakesPerWord = 5

    enum Step: Equatable {
        /// `number` is 1-based within the word.
        case take(word: String, number: Int)
        case done
        case cancelled
    }

    let words: [String]
    let takesPerWord: Int
    /// Templates recorded in this run, oldest first.
    private(set) var takes: [Template.ID] = []
    private(set) var isCancelled = false

    init(words: [String] = QuickSetup.commands, takesPerWord: Int = QuickSetup.defaultTakesPerWord) {
        precondition(!words.isEmpty && takesPerWord > 0, "Quick setup needs at least one word and one take")
        self.words = words
        self.takesPerWord = takesPerWord
    }

    var totalTakes: Int { words.count * takesPerWord }

    var step: Step {
        if isCancelled { return .cancelled }
        if takes.count >= totalTakes { return .done }
        return .take(word: words[takes.count / takesPerWord], number: takes.count % takesPerWord + 1)
    }

    var isRunning: Bool {
        if case .take = step { return true }
        return false
    }

    /// The word to prompt for, or nil once the run is done or cancelled.
    var currentWord: String? {
        if case let .take(word, _) = step { return word }
        return nil
    }

    /// Fraction of all takes recorded, 0...1.
    var progress: Double { Double(takes.count) / Double(totalTakes) }

    /// Counts a captured take for the current word and moves on. Returns that word,
    /// or nil (and records nothing) when the run is done or cancelled.
    @discardableResult
    mutating func accept(_ id: Template.ID) -> String? {
        guard let word = currentWord else { return nil }
        takes.append(id)
        return word
    }

    /// Drops the last take so it is prompted again, and returns its id for the caller to delete.
    /// Works after the last take too, which reopens the run. Does nothing once cancelled.
    mutating func redo() -> Template.ID? {
        guard !isCancelled else { return nil }
        return takes.popLast()
    }

    /// Ends the run early and returns every take recorded in it, so the caller can delete them.
    mutating func cancel() -> [Template.ID] {
        guard isRunning else { return [] }
        isCancelled = true
        let recorded = takes
        takes = []
        return recorded
    }
}
