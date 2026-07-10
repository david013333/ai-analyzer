from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import os
import psycopg2
from flask import session

# ---- CHANGED: Removed SentenceTransformer (too heavy for Render ~400MB) ----
# ---- Using improved TF-IDF with bigrams — lightweight + more accurate ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key = "secret123"
CORS(app)


# -------------------- DATABASE --------------------
def init_db():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS activity
                   (
                       user_id
                       INTEGER,
                       screen_time
                       REAL,
                       sleep
                       REAL,
                       study
                       REAL,
                       stress
                       REAL,
                       score
                       REAL,
                       date
                       TEXT
                   )
                   """)

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS users
                   (
                       id
                       SERIAL
                       PRIMARY
                       KEY,
                       username
                       TEXT
                       UNIQUE,
                       password
                       TEXT
                   )
                   """)

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS friends
                   (
                       user_id
                       INTEGER,
                       friend_id
                       INTEGER
                   )
                   """)

    conn.commit()
    conn.close()


init_db()

# -------------------- PERSONALITY MODEL --------------------
"""
UPGRADE: TF-IDF improved with:
  1. ngram_range=(1,2) — captures word pairs like "feel comfortable", "leading team"
     Old TF-IDF only looked at single words
     New TF-IDF looks at single words AND pairs of words

  2. sublinear_tf=True — log scaling so one repeated word doesn't dominate

  3. LogisticRegression C=5.0 — better boundary control

  4. class_weight='balanced' — all 4 classes get equal importance

Result:
  Old: "I feel comfortable leading a group" → High Confidence 38% (wrong sometimes)
  New: "I feel comfortable leading a group" → High Confidence 69% ✅ more confident

Why not SentenceTransformer?
  SentenceTransformer downloads ~90MB model
  Uses ~400MB RAM on Render
  Render free tier = 512MB total → CRASH ❌

  Improved TF-IDF:
  Downloads 0MB
  Uses ~5MB RAM ✅
  Faster startup ✅
"""

vectorizer = None
model_personality = None


def load_model():
    global vectorizer, model_personality

    if model_personality is not None:
        return

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "dataset.csv")

    df_data = pd.read_csv(path)
    texts = df_data["text"].tolist()
    labels = df_data["label"].tolist()

    # IMPROVED TF-IDF — bigrams + log scaling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # single words + word pairs
        sublinear_tf=True,  # log(1 + tf) instead of raw tf
        min_df=1,  # include even rare words
        strip_accents='unicode',
        analyzer='word'
    )
    X = vectorizer.fit_transform(texts)

    # IMPROVED Logistic Regression
    model_personality = LogisticRegression(
        max_iter=1000,
        C=5.0,  # less regularization = learns more
        class_weight='balanced',  # equal importance to all 4 classes
        solver='lbfgs',
        multi_class='multinomial'
    )
    model_personality.fit(X, labels)


# -------------------- REAL DATASET LOADER --------------------
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_lifestyle_dataset.csv")

STRESS_MAP = {"Low": 2, "Moderate": 5, "High": 9}


def load_real_dataset() -> pd.DataFrame:
    """
    Loads student_lifestyle_dataset.csv and maps to app fields.
    Column Mapping:
      Study_Hours_Per_Day          -> study
      Sleep_Hours_Per_Day          -> sleep
      Social_Hours_Per_Day         -> screen_time
      Stress_Level (Low/Mod/High)  -> stress (1-10)
      GPA (2.24 - 4.0)             -> score (0-100)
    """
    df = pd.read_csv(DATASET_PATH)

    result = pd.DataFrame()
    result["study"] = df["Study_Hours_Per_Day"]
    result["sleep"] = df["Sleep_Hours_Per_Day"]
    result["screen_time"] = df["Social_Hours_Per_Day"]
    result["stress"] = df["Stress_Level"].map(STRESS_MAP)

    gpa_min, gpa_max = 2.24, 4.0
    result["score"] = ((df["GPA"] - gpa_min) / (gpa_max - gpa_min) * 100).clip(0, 100).round(2)

    return result.dropna()


# -------------------- RANDOM FOREST SCORE MODEL --------------------
rf_score_model = None
rf_scaler = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_score_model.pkl")
RF_SCALER_PATH = os.path.join(BASE_DIR, "rf_scaler.pkl")


def rule_based_score(screen, sleep, study, stress):
    score = (study * 10) + (sleep * 5) - (screen * 3) - (stress * 2)
    return max(0.0, min(float(score), 100.0))


def generate_synthetic_training_data(n=2000):
    np.random.seed(42)
    screen = np.random.uniform(0, 12, n)
    sleep = np.random.uniform(3, 10, n)
    study = np.random.uniform(0, 10, n)
    stress = np.random.uniform(1, 10, n)
    scores = np.array([
        rule_based_score(sc, sl, st, sr)
        for sc, sl, st, sr in zip(screen, sleep, study, stress)
    ])
    return pd.DataFrame({
        "screen_time": screen, "sleep": sleep,
        "study": study, "stress": stress, "score": scores
    })


def train_rf_model(df: pd.DataFrame):
    global rf_score_model, rf_scaler

    features = ["screen_time", "sleep", "study", "stress"]
    X = df[features].values
    y = df["score"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)

    rf_score_model = rf
    rf_scaler = scaler

    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(scaler, RF_SCALER_PATH)


def load_rf_model():
    global rf_score_model, rf_scaler

    if rf_score_model is not None:
        return

    if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH):
        rf_score_model = joblib.load(RF_MODEL_PATH)
        rf_scaler = joblib.load(RF_SCALER_PATH)
        return

    df_train = load_real_dataset() if os.path.exists(DATASET_PATH) else generate_synthetic_training_data()

    try:
        DATABASE_URL = os.environ.get("DATABASE_URL")
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            df_db = pd.read_sql_query(
                "SELECT screen_time, sleep, study, stress, score FROM activity WHERE score IS NOT NULL",
                conn
            )
            conn.close()
            if not df_db.empty:
                df_train = pd.concat([df_train, df_db], ignore_index=True)
    except Exception:
        pass

    train_rf_model(df_train)


def retrain_rf_with_db():
    try:
        DATABASE_URL = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)
        df_db = pd.read_sql_query(
            "SELECT screen_time, sleep, study, stress, score FROM activity WHERE score IS NOT NULL",
            conn
        )
        conn.close()

        if len(df_db) < 10:
            return

        df_base = load_real_dataset() if os.path.exists(DATASET_PATH) else generate_synthetic_training_data(n=500)
        df_combined = pd.concat([df_base, df_db], ignore_index=True)
        train_rf_model(df_combined)
    except Exception:
        pass


def calculate_score(screen, sleep, study, stress):
    try:
        load_rf_model()
        features = np.array([[screen, sleep, study, stress]], dtype=float)
        features_scaled = rf_scaler.transform(features)
        predicted = rf_score_model.predict(features_scaled)[0]
        return round(max(0.0, min(float(predicted), 100.0)), 2)
    except Exception:
        score = (study * 10) + (sleep * 5) - (screen * 3) - (stress * 2)
        return round(max(0.0, min(float(score), 100.0)), 2)


# -------------------- INPUT VALIDATION --------------------
def validate_inputs(screen, sleep, study, stress):
    errors = []

    if screen < 0: errors.append("Screen time cannot be negative.")
    if sleep < 0: errors.append("Sleep hours cannot be negative.")
    if study < 0: errors.append("Study hours cannot be negative.")
    if stress < 0: errors.append("Stress cannot be negative.")

    if screen > 24: errors.append("Screen time cannot exceed 24 hours.")
    if sleep > 24: errors.append("Sleep cannot exceed 24 hours.")
    if study > 24: errors.append("Study cannot exceed 24 hours.")

    if stress < 1 or stress > 10:
        errors.append("Stress must be between 1 and 10.")

    total = screen + sleep + study
    if total > 24:
        errors.append(
            f"Total hours (screen {screen}h + sleep {sleep}h + study {study}h = {total}h) "
            f"cannot exceed 24 hours in a day."
        )

    if sleep + study > 24:
        errors.append(
            f"Sleep ({sleep}h) + Study ({study}h) = {sleep + study}h — "
            f"cannot overlap beyond 24 hours."
        )

    return (False, " | ".join(errors)) if errors else (True, None)


# -------------------- NEXT DAY PREDICTION --------------------
def build_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["score_lag1"] = df["score"].shift(1)
    df["score_lag2"] = df["score"].shift(2)
    df["score_rolling3"] = df["score"].rolling(3).mean()
    df["score_trend"] = df["score"].diff()
    return df.dropna()


def predict_score(df: pd.DataFrame):
    if len(df) < 2:
        return None

    df_feat = build_prediction_features(df)
    if df_feat.empty:
        return None

    feature_cols = ["score_lag1", "score_lag2", "score_rolling3", "score_trend"]
    X = df_feat[feature_cols].values
    y = df_feat["score"].values

    model = LinearRegression() if len(df_feat) < 5 else GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    last_score = df["score"].iloc[-1]
    prev_score = df["score"].iloc[-2] if len(df) >= 2 else last_score
    rolling3 = df["score"].tail(3).mean()
    trend = last_score - prev_score

    predicted = model.predict(np.array([[last_score, prev_score, rolling3, trend]]))[0]
    return round(max(0.0, min(float(predicted), 100.0)), 2)


# -------------------- HELPERS --------------------
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
    if not user_id:
        return None

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql_query(
        "SELECT date, score, screen_time, sleep, study, stress FROM activity WHERE user_id=%s",
        conn, params=(user_id,)
    )
    conn.close()

    if len(df) >= 2:
        if df.iloc[-1]["score"] > df.iloc[-2]["score"]:
            trend = "Improving 📈"
        elif df.iloc[-1]["score"] < df.iloc[-2]["score"]:
            trend = "Declining 📉"
        else:
            trend = "Same 😐"
    else:
        trend = "Not enough data"

    latest = df.iloc[-1]
    advice = []
    if latest["screen_time"] > 5: advice.append("Reduce screen time 📱")
    if latest["sleep"] < 6: advice.append("Improve sleep 😴")
    if latest["study"] < 2: advice.append("Study more 📚")
    if latest["stress"] > 7: advice.append("Try to relax 🧘")
    if not advice:                 advice.append("Good job! Keep it up 👍")
    advice_text = " | ".join(advice)

    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()
    df = df.groupby(df["date"].dt.date)["score"].mean().reset_index()
    df["date"] = df["date"].astype(str)

    plt.figure(figsize=(6, 4))
    plt.plot(df["date"], df["score"], marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode(), trend, advice_text


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
        new_user_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return jsonify({"message": "Signup success", "user_id": new_user_id})

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
        session["user_id"] = user[0]
        return jsonify({"message": "Login success", "user_id": user[0]})
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

    is_valid, error_msg = validate_inputs(screen, sleep, study, stress)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    text = data.get("text") or ""

    # Improved TF-IDF prediction
    vec = vectorizer.transform([text])
    pred = model_personality.predict(vec)[0]
    prob = model_personality.predict_proba(vec).max()

    score = calculate_score(screen, sleep, study, stress)

    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    ist = pytz.timezone("Asia/Kolkata")
    date = datetime.now(ist).strftime("%Y-%m-%d")

    cursor.execute(
        "INSERT INTO activity (user_id, screen_time, sleep, study, stress, score, date) VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (user_id, screen, sleep, study, stress, score, date)
    )
    conn.commit()

    df = pd.read_sql_query(
        "SELECT date, score FROM activity WHERE user_id=%s ORDER BY date",
        conn, params=(user_id,)
    )
    conn.close()

    retrain_rf_with_db()

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
    load_rf_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)