import SafariServices
import SwiftUI

/// The extension target's bundle identifier is the app's plus ".Extension".
private let extensionIdentifier = (Bundle.main.bundleIdentifier ?? "ax.hv.ingest") + ".Extension"

private enum Palette {
    static let background = Color(red: 0x14 / 255, green: 0x12 / 255, blue: 0x0f / 255)
    static let foreground = Color(red: 0xec / 255, green: 0xe6 / 255, blue: 0xdc / 255)
    static let muted = Color(red: 0x9a / 255, green: 0x90 / 255, blue: 0x83 / 255)
    static let line = Color(red: 0x2c / 255, green: 0x28 / 255, blue: 0x22 / 255)
    static let accent = Color(red: 0xd2 / 255, green: 0x6a / 255, blue: 0x2c / 255)
}

private enum ExtensionState {
    case checking
    case enabled
    case disabled
    case unavailable(String)
}

@main
struct HvaxIngestApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("hvax ingest", id: "main") {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

private struct ContentView: View {
    @State private var state: ExtensionState = .checking

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(nsImage: NSApp.applicationIconImage)
                    .resizable()
                    .frame(width: 48, height: 48)
                VStack(alignment: .leading, spacing: 2) {
                    Text("hvax ingest")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Palette.foreground)
                    statusLine
                }
            }

            Text(
                """
                The extension runs inside Safari. Turn it on in Safari Settings → \
                Extensions, then choose “Always Allow on Every Website” so it can \
                read page images and reach your gallery.
                """
            )
            .font(.callout)
            .foregroundStyle(Palette.muted)
            .fixedSize(horizontal: false, vertical: true)

            Button("Open Safari Extensions Settings…", action: openSafariSettings)
                .buttonStyle(.borderedProminent)
                .tint(Palette.accent)
                .keyboardShortcut(.defaultAction)
        }
        .padding(20)
        .frame(width: 380, alignment: .leading)
        .background(Palette.background)
        .preferredColorScheme(.dark)
        .onAppear(perform: refresh)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refresh()
        }
    }

    @ViewBuilder private var statusLine: some View {
        switch state {
        case .checking:
            Text("checking Safari…").foregroundStyle(Palette.muted)
        case .enabled:
            Label("on in Safari", systemImage: "checkmark.circle.fill").foregroundStyle(Palette.accent)
        case .disabled:
            Label("off in Safari", systemImage: "circle").foregroundStyle(Palette.muted)
        case .unavailable(let reason):
            Label(reason, systemImage: "exclamationmark.triangle").foregroundStyle(Palette.muted)
        }
    }

    private func refresh() {
        SFSafariExtensionManager.getStateOfSafariExtension(withIdentifier: extensionIdentifier) { extensionState, error in
            let next: ExtensionState
            if let extensionState {
                next = extensionState.isEnabled ? .enabled : .disabled
            } else {
                next = .unavailable(error?.localizedDescription ?? "Safari did not report the extension")
            }
            DispatchQueue.main.async {
                state = next
            }
        }
    }

    private func openSafariSettings() {
        SFSafariApplication.showPreferencesForExtension(withIdentifier: extensionIdentifier) { _ in }
    }
}
