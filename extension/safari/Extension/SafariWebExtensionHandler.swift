import SafariServices

/// Safari instantiates this class for `browser.runtime.sendNativeMessage`.
/// The extension does all of its work in JavaScript and never sends one, so
/// any request is completed with an empty reply.
final class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {
    func beginRequest(with context: NSExtensionContext) {
        context.completeRequest(returningItems: [NSExtensionItem()], completionHandler: nil)
    }
}
