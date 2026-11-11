import XCTest
@testable import Subvox

final class PhononPathTests: XCTestCase {
    private var defaults: UserDefaults!
    private let suite = "PhononPathTests"

    override func setUp() {
        defaults = UserDefaults(suiteName: suite)
        defaults.removePersistentDomain(forName: suite)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    private func resolve(env: String? = nil, stored: String? = nil, executable: Set<String>) -> String? {
        defaults.set(stored, forKey: "phononPath")
        let environment = env.map { ["SUBVOX_PHONON": $0] } ?? [:]
        return SessionRecorder.phononURL(environment: environment, defaults: defaults,
                                         isExecutable: { executable.contains($0) })?.path
    }

    func testEnvironmentBeatsDefaultsBeatsDevPath() {
        let dev = SessionRecorder.devPhonon.path
        let all: Set = ["/env/phonon", "/stored/phonon", dev]
        XCTAssertEqual(resolve(env: "/env/phonon", stored: "/stored/phonon", executable: all), "/env/phonon")
        XCTAssertEqual(resolve(stored: "/stored/phonon", executable: all), "/stored/phonon")
        XCTAssertEqual(resolve(executable: all), dev)
    }

    func testSkipsPathsThatAreNotExecutable() {
        let dev = SessionRecorder.devPhonon.path
        XCTAssertEqual(resolve(env: "/missing", stored: "/stored/phonon", executable: ["/stored/phonon", dev]), "/stored/phonon")
        XCTAssertEqual(resolve(env: "", stored: "/missing", executable: [dev]), dev)
        XCTAssertNil(resolve(env: "/missing", stored: "/missing", executable: []))
    }
}
