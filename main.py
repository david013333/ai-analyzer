from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime

app = Flask(__name__)
CORS(app)

# -------------------- DATABASE --------------------
def init_db():
    conn = sqlite3.connect("project.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity (
        user_id INTEGER,
        screen_time REAL,
        sleep REAL,
        study REAL,
        stress REAL,
        score REAL,
        date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS friends (
        user_id INTEGER,
        friend_id INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()

# -------------------- ML MODEL (LAZY LOAD) --------------------
model_personality = None
vectorizer = None

def load_model():
    global model_personality, vectorizer

    if model_personality is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE_DIR, "dataset.csv")

        df_data = pd.read_csv(path)

        texts = df_data["text"]
        labels = df_data["label"]

        vectorizer = TfidfVectorizer()
        X_text = vectorizer.fit_transform(texts)

        model_personality = LogisticRegression(max_iter=1000)
        model_personality.fit(X_text, labels)

# -------------------- HELPERS --------------------
def calculate_score(screen, sleep, study, stress):
    score = (study * 10) + (sleep * 5) - (screen * 3) - (stress * 2)
    return max(0, min(score, 100))

def give_suggestion(personality, score, progress):
    suggestions = {
        "Introvert": "Try socializing a bit every day.",
        "Extrovert": "Use your energy productively.",
        "Low Confidence": "Start journaling.",
        "High Confidence": "Help others grow."
    }

    if progress > 0:
        msg = f" Improving (+{progress:.1f})"
    elif progress < 0:
        msg = f" Dropped (-{abs(progress):.1f})"
    else:
        msg = " Stable"

    return suggestions.get(personality, "") + msg

def generate_plot(user_id):
    conn = sqlite3.connect("project.db")
    df = pd.read_sql_query(
        f"SELECT date, score FROM activity WHERE user_id={user_id}", conn)
    conn.close()

    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna()

    df = df.groupby(df["date"].dt.date)["score"].mean().reset_index()
    df["date"] = df["date"].astype(str)

    plt.figure(figsize=(6,4))
    plt.plot(df["date"], df["score"], marker='o')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_b64

# -------------------- AUTH --------------------
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json(force=True)

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"message": "Missing fields"}), 400

        conn = sqlite3.connect("project.db")
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )

        conn.commit()
        conn.close()

        return jsonify({"message": "Signup success"})

    except sqlite3.IntegrityError:
        return jsonify({"message": "User exists"})

    except Exception as e:
        return jsonify({"message": "Server error", "error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json(force=True)

        username = data.get("username")
        password = data.get("password")

        conn = sqlite3.connect("project.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({"message": "Login success", "user_id": user[0]})
        else:
            return jsonify({"message": "Login failed"})

    except Exception as e:
        return jsonify({"message": "Server error", "error": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/reset_db")
def reset_db():
    conn = sqlite3.connect("project.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS activity")
    cursor.execute("DROP TABLE IF EXISTS friends")

    conn.commit()
    conn.close()

    init_db()

    return "DB Reset Done"

# -------------------- FRIEND --------------------
@app.route("/add_friend", methods=["POST"])
def add_friend():
    data = request.get_json(force=True)

    conn = sqlite3.connect("project.db")
    cursor = conn.cursor()

    cursor.execute("INSERT INTO friends VALUES (?, ?)",
                   (data["user_id"], data["friend_id"]))

    conn.commit()
    conn.close()

    return jsonify({"message": "Friend added"})

# -------------------- LEADERBOARD --------------------
@app.route("/leaderboard", methods=["POST"])
def leaderboard():
    user_id = request.get_json(force=True)["user_id"]

    conn = sqlite3.connect("project.db")

    query = f"""
    SELECT users.username, MAX(activity.score) as score
    FROM activity
    JOIN users ON users.id = activity.user_id
    WHERE activity.user_id IN (
        SELECT friend_id FROM friends WHERE user_id = {user_id}
    )
    GROUP BY users.username
    ORDER BY score DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df.to_json(orient="records")

# -------------------- ANALYZE --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    load_model()

    user_id = data["user_id"]
    screen = float(data["screen_time"])
    sleep = float(data["sleep"])
    study = float(data["study"])
    stress = float(data["stress"])
    text = data["text"]

    vec = vectorizer.transform([text])
    pred = model_personality.predict(vec)[0]
    prob = model_personality.predict_proba(vec).max()

    score = calculate_score(screen, sleep, study, stress)

    conn = sqlite3.connect("project.db")
    cursor = conn.cursor()

    cursor.execute("INSERT INTO activity VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (user_id, screen, sleep, study, stress, score,
                    datetime.now().strftime("%Y-%m-%d")))
    conn.commit()

    df = pd.read_sql_query(
        f"SELECT score FROM activity WHERE user_id={user_id}", conn)
    conn.close()

    progress = 0
    if len(df) >= 2:
        progress = score - df.iloc[-2]["score"]

    graph = generate_plot(user_id)

    return jsonify({
        "personality": pred,
        "confidence": round(prob * 100, 2),
        "score": score,
        "progress": progress,
        "suggestion": give_suggestion(pred, score, progress),
        "graph": f"data:image/png;base64,{graph}" if graph else None
    })

# -------------------- HOME --------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)