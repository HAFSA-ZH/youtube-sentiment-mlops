
console.log("YouTube Sentiment Analyzer : content.js chargé");

// Sélecteurs des commentaires YouTube
const COMMENT_SELECTORS = [
    "ytd-comment-view-model #content-text",
    "ytd-comment-renderer #content-text",
    "ytd-comment-thread-renderer #content-text"
];

// Vérifier que le commentaire est affiché
function isVisible(element) {

    const style = window.getComputedStyle(element);

    return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        element.getClientRects().length > 0
    );
}

// Extraire les commentaires
function extractComments() {

    const selector = COMMENT_SELECTORS.join(",");

    const elements = document.querySelectorAll(selector);

    const comments = [];
    const seen = new Set();

    elements.forEach(element => {

        if (!isVisible(element)) {
            return;
        }

        const text = element.innerText
            .trim()
            .replace(/\s+/g, " ");

        if (!text || seen.has(text)) {
            return;
        }

        seen.add(text);
        comments.push(text);

    });

    // Notre API accepte au maximum 100 commentaires par requête
    return comments.slice(0, 100);
}

// Recevoir une demande de l'extension
chrome.runtime.onMessage.addListener(
    (message, sender, sendResponse) => {

        if (message.action === "EXTRACT_COMMENTS") {

            const comments = extractComments();

            console.log(
                "Commentaires extraits :",
                comments.length
            );

            sendResponse({
                success: true,
                comments: comments
            });
        }

    }
);
