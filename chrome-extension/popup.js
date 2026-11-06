
const API_URL = "http://127.0.0.1:8000";

const LABELS = {
    negative: "Négatif",
    neutral: "Neutre",
    positive: "Positif"
};

let predictions = [];
let currentFilter = "all";

const analyzeBtn = document.getElementById("analyzeBtn");
const message = document.getElementById("message");
const resultsSection = document.getElementById("results");
const resultsList = document.getElementById("resultsList");

// =====================================
// 1. GESTION DES MESSAGES
// =====================================

function showMessage(text, type = "info") {

    message.textContent = text;
    message.className = "message " + type;
}

function setApiStatus(connected) {

    const dot = document.getElementById("statusDot");
    const status = document.getElementById("apiStatus");

    dot.className = connected
        ? "status-dot connected"
        : "status-dot disconnected";

    status.textContent = connected
        ? "API connectée"
        : "API non disponible";
}

// =====================================
// 2. VERIFIER L'API
// =====================================

async function checkAPI() {

    try {

        const response = await fetch(
            `${API_URL}/health`
        );

        if (!response.ok) {
            throw new Error("API indisponible");
        }

        const data = await response.json();

        setApiStatus(data.status === "ok");

    } catch (error) {

        setApiStatus(false);

    }
}

// =====================================
// 3. EXTRAIRE LES COMMENTAIRES
// =====================================

async function extractComments() {

    const tabs = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    const tab = tabs[0];

    if (!tab || !tab.url ||
        !tab.url.includes("youtube.com")) {

        throw new Error(
            "Ouvre une vidéo YouTube avant l'analyse."
        );
    }

    try {

        const response = await chrome.tabs.sendMessage(
            tab.id,
            {
                action: "EXTRACT_COMMENTS"
            }
        );

        return response.comments || [];

    } catch (error) {

        throw new Error(
            "Recharge la page YouTube puis réessaie."
        );
    }
}

// =====================================
// 4. ENVOYER LES COMMENTAIRES A FASTAPI
// =====================================

async function analyzeComments(comments) {

    const response = await fetch(
        `${API_URL}/predict_batch`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                comments: comments.slice(0, 100)
            })
        }
    );

    if (!response.ok) {

        const error = await response.json().catch(
            () => ({ detail: "Erreur API" })
        );

        throw new Error(
            error.detail || "Erreur pendant l'analyse."
        );
    }

    return await response.json();
}

// =====================================
// 5. AFFICHER LES STATISTIQUES
// =====================================

function updateStatistics(statistics) {

    const sentiments = [
        "negative",
        "neutral",
        "positive"
    ];

    sentiments.forEach(sentiment => {

        document.getElementById(
            sentiment + "Count"
        ).textContent = statistics.counts[sentiment];

        document.getElementById(
            sentiment + "Percent"
        ).textContent =
            statistics.percentages[sentiment] + "%";

    });

    document.getElementById(
        "totalComments"
    ).textContent =
        statistics.total + " commentaires";
}

// =====================================
// 6. AFFICHER UNE PREDICTION
// =====================================

function createResultCard(item) {

    const card = document.createElement("article");
    card.className = "result-card";

    const top = document.createElement("div");
    top.className = "result-top";

    const badge = document.createElement("span");

    badge.className =
        "sentiment-badge " + item.sentiment;

    badge.textContent = LABELS[item.sentiment];

    const confidence = document.createElement("span");

    confidence.className = "confidence";

    confidence.textContent =
        "Confiance : " +
        (item.confidence * 100).toFixed(1) + "%";

    top.append(badge, confidence);

    const text = document.createElement("p");

    text.className = "comment-text";
    text.textContent = item.comment;

    const track = document.createElement("div");
    track.className = "confidence-track";

    const fill = document.createElement("div");
    fill.className = "confidence-fill";

    const confidencePercent = Math.min(
        100,
        Math.max(0, item.confidence * 100)
    );

    fill.style.width = confidencePercent + "%";

    track.appendChild(fill);

    card.append(top, text, track);

    return card;
}

// =====================================
// 7. AFFICHER LA LISTE
// =====================================

function renderResults() {

    resultsList.replaceChildren();

    const filtered = predictions.filter(item => {

        return (
            currentFilter === "all" ||
            item.sentiment === currentFilter
        );

    });

    if (filtered.length === 0) {

        const empty = document.createElement("p");

        empty.className = "message";
        empty.textContent = "Aucun commentaire dans ce filtre.";

        resultsList.appendChild(empty);

        return;
    }

    filtered.forEach(item => {

        resultsList.appendChild(
            createResultCard(item)
        );

    });
}

// =====================================
// 8. BOUTON ANALYSER
// =====================================

analyzeBtn.addEventListener("click", async () => {

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyse en cours...";

    try {

        showMessage(
            "Extraction des commentaires YouTube..."
        );

        const comments = await extractComments();

        if (comments.length === 0) {

            throw new Error(
                "Aucun commentaire détecté. "
                + "Descends jusqu'aux commentaires "
                + "et charge-les en faisant défiler la page."
            );
        }

        showMessage(
            comments.length + " commentaires extraits. Analyse IA..."
        );

        const data = await analyzeComments(comments);

        predictions = data.predictions;

        currentFilter = "all";

        document.querySelectorAll(".filter").forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.filter === "all"
            );

        });

        updateStatistics(data.statistics);

        renderResults();

        resultsSection.hidden = false;

        document.getElementById(
            "inferenceTime"
        ).textContent =
            "Temps d'analyse : " +
            data.inference_time_ms + " ms";

        showMessage(
            "Analyse terminée avec succès !",
            "success"
        );

    } catch (error) {

        showMessage(error.message, "error");

    } finally {

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML =
            "<span>✦</span> Analyser les commentaires";

    }

});

// =====================================
// 9. FILTRES
// =====================================

document.querySelectorAll(".filter").forEach(button => {

    button.addEventListener("click", () => {

        currentFilter = button.dataset.filter;

        document.querySelectorAll(".filter").forEach(btn => {
            btn.classList.remove("active");
        });

        button.classList.add("active");

        renderResults();

    });

});

// =====================================
// 10. COPIER LES RESULTATS
// =====================================

document.getElementById("copyBtn").addEventListener(
    "click",
    async () => {

        const filtered = predictions.filter(item => {

            return (
                currentFilter === "all" ||
                item.sentiment === currentFilter
            );

        });

        const text = filtered.map(item => {

            return (
                item.comment +
                " | " +
                LABELS[item.sentiment] +
                " | " +
                (item.confidence * 100).toFixed(1) +
                "%"
            );

        }).join("\n\n");

        try {

            await navigator.clipboard.writeText(text);

            showMessage(
                "Résultats copiés avec succès !",
                "success"
            );

        } catch (error) {

            showMessage(
                "Impossible de copier les résultats.",
                "error"
            );

        }

    }
);

// =====================================
// 11. MODE SOMBRE / CLAIR
// =====================================

const themeToggle = document.getElementById("themeToggle");

const savedTheme = localStorage.getItem("theme");

if (savedTheme === "dark") {

    document.body.classList.add("dark");
    themeToggle.textContent = "☀";

}

themeToggle.addEventListener("click", () => {

    document.body.classList.toggle("dark");

    const darkMode = document.body.classList.contains("dark");

    localStorage.setItem(
        "theme",
        darkMode ? "dark" : "light"
    );

    themeToggle.textContent = darkMode ? "☀" : "☾";

});

// =====================================
// 12. INITIALISATION
// =====================================

checkAPI();
