from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
import google.generativeai as genai
import datetime
import re
import json
import os
from werkzeug.utils import secure_filename
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Google Generative AI --------------------
GEMINI_API_KEY = "AIzaSyCYrkFo8DR98iAJNOdZhy9WRTJeuH2lexw"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# -------------------- Firebase --------------------
cred = credentials.Certificate("ai-study-assistant-4a267-firebase-adminsdk-fbsvc-8797041d6a.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- Helper: Verify Firebase Token --------------------
def verify_token(req):
    auth_header = req.headers.get("Authorization")
    if not auth_header:
        return None
    token = auth_header.split("Bearer ")[-1]
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded["uid"]  # Firebase User ID
    except Exception:
        return None

# ---------- Helpers ----------

def extract_text_from_file(filepath):
    text = ""
    if filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif filepath.lower().endswith(".pdf"):
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    return text or ""

def safe_clean(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_year_from_filename(fname):
    # find first 4-digit year 1900-2099 in filename
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        try:
            return int(m.group(0))
        except:
            return None
    return None

# ---- Improved syllabus/topic splitting ----

def split_syllabus(syllabus_text):
    """
    Extract syllabus topics from raw text.
    - Splits by newline or bullet points (NOT commas inside sentences).
    - Filters out too-short items.
    """
    parts = re.split(r'[\r\n•\-–]+', syllabus_text)
    topics = [p.strip() for p in parts if p and len(p.strip()) > 3]
    return topics

STOPWORDS = {
    "in", "on", "at", "the", "a", "an", "and", "or", "of",
    "to", "for", "by", "is", "are", "be", "write"
}

def filter_topics(topics):
    """Remove too-short and stopword-only topics"""
    filtered = []
    for t in topics:
        words = t.split()
        if len(words) == 0:
            continue
        if len(words) == 1 and (words[0] in STOPWORDS or words[0].isdigit()):
            continue
        filtered.append(t)
    return filtered

def fuzzy_similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# ---------- Core analysis (multi-signal) ----------

def compute_predictions(docs, syllabus_topics):
    current_year = datetime.datetime.now().year
    n_topics = len(syllabus_topics)
    if n_topics == 0:
        return {}

    doc_texts = [d['text'] if d['text'] else "" for d in docs]
    if len(doc_texts) == 0:
        doc_texts = [""]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
    except Exception:
        vectorizer = None
        tfidf_matrix = None

    freq_scores, tfidf_scores, recency_scores, trend_scores, fuzzy_scores = [], [], [], [], []

    for topic in syllabus_topics:
        topic_clean = topic.strip()
        if not topic_clean:
            freq_scores.append(0.0)
            tfidf_scores.append(0.0)
            recency_scores.append(0.0)
            trend_scores.append(0.0)
            fuzzy_scores.append(0.0)
            continue

        pattern = re.compile(rf"\b{re.escape(topic_clean)}\b", flags=re.IGNORECASE)
        counts_per_doc = []
        year_counts = defaultdict(int)

        for idx, d in enumerate(docs):
            ct = len(pattern.findall(d['text'])) if d['text'] else 0
            counts_per_doc.append(ct)
            if d.get('year'):
                year_counts[d['year']] += ct

        freq_score = float(sum(counts_per_doc))

        tfidf_score = 0.0
        if vectorizer is not None:
            try:
                topic_vec = vectorizer.transform([topic_clean])
                relevance = (tfidf_matrix @ topic_vec.T).toarray().flatten()
                tfidf_score = float(relevance.sum())
            except Exception:
                tfidf_score = 0.0

        recency_score = 0.0
        for idx, d in enumerate(docs):
            y = d.get('year')
            ct = counts_per_doc[idx]
            if ct <= 0:
                continue
            if y and isinstance(y, int):
                age = max(0, current_year - y)
                weight = 1.0 / (age + 1.0)
            else:
                weight = 0.8
            recency_score += ct * weight

        trend_score = 0.0
        if len(year_counts) >= 2:
            years = sorted(year_counts.keys())
            counts = [year_counts[y] for y in years]
            try:
                coef = np.polyfit(years, counts, 1)
                slope = float(coef[0])
                trend_score = slope if slope > 0 else 0.0
            except Exception:
                trend_score = 0.0

        max_fuzzy = 0.0
        for d in docs:
            snippet = (d['text'][:300]) if d['text'] else ''
            name = d.get('filename', '')
            r1 = fuzzy_similarity(topic_clean, snippet)
            r2 = fuzzy_similarity(topic_clean, name)
            max_fuzzy = max(max_fuzzy, r1, r2)
        fuzzy_score = max_fuzzy

        freq_scores.append(freq_score)
        tfidf_scores.append(tfidf_score)
        recency_scores.append(recency_score)
        trend_scores.append(trend_score)
        fuzzy_scores.append(fuzzy_score)

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        maxv = arr.max() if arr.size > 0 else 0.0
        return arr / maxv if maxv > 0 else arr * 0.0

    n_freq, n_tfidf, n_recency, n_trend, n_fuzzy = (
        normalize(freq_scores),
        normalize(tfidf_scores),
        normalize(recency_scores),
        normalize(trend_scores),
        normalize(fuzzy_scores),
    )

    w_freq, w_tfidf, w_recency, w_trend, w_fuzzy = 0.35, 0.30, 0.15, 0.10, 0.10

    final_raw = (
        w_freq * n_freq +
        w_tfidf * n_tfidf +
        w_recency * n_recency +
        w_trend * n_trend +
        w_fuzzy * n_fuzzy
    )

    total_raw = final_raw.sum()
    if total_raw <= 0:
        return {t: 0.0 for t in syllabus_topics}

    final_percentages = (final_raw / total_raw) * 100.0
    results = {topic: round(float(score), 2) for topic, score in zip(syllabus_topics, final_percentages)}
    results_sorted = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return results_sorted

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        syllabus_file = request.files.get("syllabus")

        if (not uploaded_files or len(uploaded_files) == 0) or not syllabus_file:
            return "Please upload both question papers and a syllabus file.", 400

        docs = []
        for file in uploaded_files:
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            raw = extract_text_from_file(filepath)
            cleaned = safe_clean(raw)
            year = extract_year_from_filename(filename)
            docs.append({"filename": filename, "year": year, "text": cleaned})

        syllabus_filename = secure_filename(syllabus_file.filename)
        syllabus_filepath = os.path.join(app.config["UPLOAD_FOLDER"], syllabus_filename)
        syllabus_file.save(syllabus_filepath)
        syllabus_raw = extract_text_from_file(syllabus_filepath)

        topics_raw = split_syllabus(syllabus_raw)
        topics_raw = filter_topics(topics_raw)   # 👈 improved filtering
        topics = [safe_clean(t) for t in topics_raw if t.strip()]

        if not topics:
            return "Syllabus file didn't contain any topics (check formatting).", 400

        try:
            results = compute_predictions(docs, topics)
        except Exception:
            results = {}
            papers_concat = " ".join(d['text'] for d in docs)
            for t in topics:
                results[t] = float(len(re.findall(rf"\b{re.escape(t)}\b", papers_concat)))
            total = sum(results.values()) if sum(results.values()) > 0 else 1
            results = {k: round((v/total)*100, 2) for k, v in results.items()}

        labels = list(results.keys())
        values = list(results.values())
        top_5 = dict(list(results.items())[:5])

        return render_template("summarize_result.html",
                               topic_percent=results,
                               labels=labels,
                               values=values,
                               top_5=top_5)

    return render_template("summarize.html")

@app.route("/quiz", methods=["GET"])
def quiz():
    return render_template("generate_quiz.html")

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    user_id = verify_token(request)
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    topic = data.get("topic", "")
    difficulty = data.get("difficulty", "easy")
    quiz_type = data.get("quiz_type", "mcq")

    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    prompt = f"""
    Generate a {difficulty} {quiz_type} quiz on the topic "{topic}".
    Return ONLY valid JSON, no markdown, no explanation.
    Format strictly like this:
    {{
      "quiz": [
        {{
          "question": "string",
          "options": ["string1", "string2", "string3", "string4"],
          "answer": 0
        }}
      ]
    }}
    Minimum 5 questions, maximum 10.
    """

    try:
        gemini_response = model.generate_content(prompt)
        raw_output = gemini_response.text.strip()
        print("Gemini raw output:", raw_output)

        if raw_output.startswith("```"):
            raw_output = re.sub(r"^```[a-zA-Z]*\n", "", raw_output)
            raw_output = raw_output.rstrip("`").rstrip()
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3].strip()

        quiz_json = json.loads(raw_output)

        quiz_ref = db.collection("users").document(user_id).collection("quizzes")
        quiz_ref.add({
            "topic": topic,
            "difficulty": difficulty,
            "quiz_type": quiz_type,
            "quiz": quiz_json.get("quiz", []),
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify(quiz_json)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "raw_output": raw_output
        }), 500

@app.route("/ask")
def ask():
    return render_template("ask.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = verify_token(request)
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    message = data.get("message")

    if not message:
        return jsonify({"error": "Missing message"}), 400

    try:
        gemini_response = model.generate_content(message)
        bot_reply = gemini_response.text

        chat_ref = db.collection("users").document(user_id).collection("chats")
        chat_ref.add({
            "message": message,
            "reply": bot_reply,
            "timestamp": datetime.datetime.utcnow()
        })

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    user_id = verify_token(request)
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        chat_ref = db.collection("users").document(user_id).collection("chats").order_by("timestamp")
        history_data = [doc.to_dict() for doc in chat_ref.stream()]
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/quiz_history", methods=["GET"])
def quiz_history():
    user_id = verify_token(request)
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        quiz_ref = (
            db.collection("users")
              .document(user_id)
              .collection("quizzes")
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(10)
        )
        history_data = []
        for doc in quiz_ref.stream():
            q = doc.to_dict()
            history_data.append({
                "topic": q.get("topic"),
                "difficulty": q.get("difficulty"),
                "quiz_type": q.get("quiz_type"),
                "timestamp": q.get("timestamp").isoformat() if q.get("timestamp") else None
            })

        return jsonify(history_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import threading, webbrowser
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, host="0.0.0.0", port=5000)
