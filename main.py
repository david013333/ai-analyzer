from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import os
import psycopg2
from flask import session

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key="secret123"
CORS(app)

# -------------------- DATABASE --------------------
def init_db():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
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
        id SERIAL primary key,
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

def predict_score(df):
    if len(df) < 2:
        return None

    last = df.iloc[-1]["score"]
    prev = df.iloc[-2]["score"]

    trend = last - prev

    return round(last + trend, 2)

def generate_plot(user_id):
    if not user_id:
        return None


    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)

    df = pd.read_sql_query(
        sql="SELECT date, score, screen_time, sleep, study, stress FROM activity WHERE user_id=%s",
        con=conn,
        params=(user_id,)
    )

    conn.close()
    if len(df) >= 2:
        today_score = df.iloc[-1]['score']
        yesterday_score = df.iloc[-2]['score']

        if today_score > yesterday_score:
            trend = "Improving 📈"
        elif today_score < yesterday_score:
            trend = "Declining 📉"
        else:
            trend = "Same 😐"
    else:
        trend = "Not enough data"

    # --- AI Advice ---
    latest = df.iloc[-1]

    advice = []

    if latest['screen_time'] > 5:
        advice.append("Reduce screen time 📱")

    if latest['sleep'] < 6:
        advice.append("Improve sleep 😴")

    if latest['study'] < 2:
        advice.append("Study more 📚")

    if latest['stress'] > 7:
        advice.append("Try to relax 🧘")

    if not advice:
        advice.append("Good job! Keep it up 👍")

    advice_text = " | ".join(advice)

    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()

    df = df.groupby(df["date"].dt.date)["score"].mean().reset_index()
    df["date"] = df["date"].astype(str)

    plt.figure(figsize=(6, 4))
    plt.plot(df["date"], df["score"], marker='o')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_b64 = base64.b64encode(img.getvalue()).decode()
    return img_b64,trend,advice_text

# -------------------- AUTH --------------------
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json(force=True)

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"message": "Missing fields"}), 400

        DATABASE_URL = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
            (username.lower(), password)
        )

        new_user_id = cursor.fetchone()[0]  # 🔥 IMPORTANT

        conn.commit()
        conn.close()

        return jsonify({"message": "Signup success","user_id": new_user_id})

    except psycopg2.errors.UniqueViolation:
        return jsonify({"message": "User exists"})

    except Exception as e:
        return jsonify({"message": "Server error", "error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    username = data.get("username").lower()
    password = data.get("password")

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE username=%s AND password=%s",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user[0]  # 🔥 IMPORTANT
        return jsonify({
            "message": "Login success",
            "user_id": user[0]  # 🔥 VERY IMPORTANT
        })
    else:
        return jsonify({"message": "Login failed"})

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/reset_db")
def reset_db():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
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

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
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

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)

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

    user_id = data.get("user_id")
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "Invalid user_id"}), 400

    screen = float(data.get("screen_time") or 0)
    sleep = float(data.get("sleep") or 0)
    study = float(data.get("study") or 0)
    stress = float(data.get("stress") or 0)

    text = data.get("text") or ""

    vec = vectorizer.transform([text])
    pred = model_personality.predict(vec)[0]
    prob = model_personality.predict_proba(vec).max()

    score = calculate_score(screen, sleep, study, stress)

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    ist = pytz.timezone('Asia/Kolkata')
    date = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO activity (user_id, screen_time, sleep, study, stress, score, date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, screen, sleep, study, stress, score, date)
    )

    conn.commit()

    df = pd.read_sql_query(
        "SELECT score FROM activity WHERE user_id=%s ORDER BY date",
        conn,
        params=(user_id,)
    )

    conn.close()

    progress = 0
    if len(df) >= 2:
        progress = score - df.iloc[-2]["score"]

    graph, trend, advice = generate_plot(user_id)
    predicted = predict_score(df)

    return jsonify({
        "personality": pred,
        "confidence": round(prob * 100, 2),
        "score": score,
        "progress": progress,
        "suggestion": give_suggestion(pred, score, progress),
        "graph": f"data:image/png;base64,{graph}" if graph else None,
        "trend": trend,
        "advice": advice,
        "prediction": predicted
    })

# -------------------- HOME --------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)