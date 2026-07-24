from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import matplotlib

matplotlib.use('Agg')  # ← faster, no GUI thread
import matplotlib.pyplot as plt
import os
import psycopg2
from flask import session
import threading

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------- DATABASE --------------------
def get_conn():
    return psycopg2.connect(os.environ.get("DATABASE_URL"))


def init_db():
    conn = get_conn()
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
                   )""")
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
                   )""")
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS friends
                   (
                       user_id
                       INTEGER,
                       friend_id
                       INTEGER
                   )""")
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS streak
                   (
                       user_id
                       INTEGER
                       UNIQUE,
                       target
                       REAL,
                       current_streak
                       INTEGER
                       DEFAULT
                       0,
                       longest_streak
                       INTEGER
                       DEFAULT
                       0,
                       last_date
                       TEXT
                   )""")
    conn.commit()
    conn.close()


init_db()

# -------------------- PERSONALITY MODEL --------------------
vectorizer = None
model_personality = None


def load_model():
    global vectorizer, model_personality
    if model_personality is not None:
        return
    path = os.path.join(BASE_DIR, "dataset.csv")
    df_data = pd.read_csv(path)
    texts = df_data["text"].tolist()
    labels = df_data["label"].tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vectorizer.fit_transform(texts)
    model_personality = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced', solver='lbfgs')
    model_personality.fit(X, labels)


def get_personality_prediction(text: str):
    vec = vectorizer.transform([text])
    pred = model_personality.predict(vec)[0]
    prob = model_personality.predict_proba(vec).max()
    return pred, prob


# -------------------- REAL DATASET --------------------
DATASET_PATH = os.path.join(BASE_DIR, "student_lifestyle_dataset.csv")
STRESS_MAP = {"Low": 2, "Moderate": 5, "High": 9}


def load_real_dataset():
    df = pd.read_csv(DATASET_PATH)
    result = pd.DataFrame()
    result["study"] = df["Study_Hours_Per_Day"]
    result["sleep"] = df["Sleep_Hours_Per_Day"]
    result["screen_time"] = df["Social_Hours_Per_Day"]
    result["stress"] = df["Stress_Level"].map(STRESS_MAP)
    gpa_min, gpa_max = 2.24, 4.0
    result["score"] = ((df["GPA"] - gpa_min) / (gpa_max - gpa_min) * 100).clip(0, 100).round(2)
    return result.dropna()


# -------------------- RANDOM FOREST --------------------
rf_score_model = None
rf_scaler = None
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
    scores = [rule_based_score(sc, sl, st, sr) for sc, sl, st, sr in zip(screen, sleep, study, stress)]
    return pd.DataFrame({"screen_time": screen, "sleep": sleep, "study": study, "stress": stress, "score": scores})


def train_rf_model(df):
    global rf_score_model, rf_scaler
    features = ["screen_time", "sleep", "study", "stress"]
    X = df[features].values
    y = df["score"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
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
        conn = get_conn()
        df_db = pd.read_sql_query(
            "SELECT screen_time, sleep, study, stress, score FROM activity WHERE score IS NOT NULL", conn)
        conn.close()
        if not df_db.empty:
            df_train = pd.concat([df_train, df_db], ignore_index=True)
    except Exception:
        pass
    train_rf_model(df_train)


def retrain_rf_with_db():
    try:
        conn = get_conn()
        df_db = pd.read_sql_query(
            "SELECT screen_time, sleep, study, stress, score FROM activity WHERE score IS NOT NULL", conn)
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


# -------------------- VALIDATION --------------------
def validate_inputs(screen, sleep, study, stress):
    errors = []
    if screen < 0:  errors.append("Screen time cannot be negative.")
    if sleep < 0:  errors.append("Sleep hours cannot be negative.")
    if study < 0:  errors.append("Study hours cannot be negative.")
    if stress < 0:  errors.append("Stress cannot be negative.")
    if screen > 24: errors.append("Screen time cannot exceed 24 hours.")
    if sleep > 24: errors.append("Sleep cannot exceed 24 hours.")
    if study > 24: errors.append("Study cannot exceed 24 hours.")
    if stress < 1 or stress > 10: errors.append("Stress must be between 1 and 10.")
    total = screen + sleep + study
    if total > 24:
        errors.append(f"Total hours ({screen}h+{sleep}h+{study}h={total}h) cannot exceed 24.")
    if sleep + study > 24:
        errors.append(f"Sleep ({sleep}h) + Study ({study}h) cannot overlap beyond 24 hours.")
    return (False, " | ".join(errors)) if errors else (True, None)


# -------------------- PREDICTION --------------------
def build_prediction_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["score_lag1"] = df["score"].shift(1)
    df["score_lag2"] = df["score"].shift(2)
    df["score_rolling3"] = df["score"].rolling(3).mean()
    df["score_trend"] = df["score"].diff()
    return df.dropna()


def predict_score(df):
    if len(df) < 2:
        return None
    df_feat = build_prediction_features(df)
    if df_feat.empty:
        return None
    feature_cols = ["score_lag1", "score_lag2", "score_rolling3", "score_trend"]
    X = df_feat[feature_cols].values
    y = df_feat["score"].values
    model = LinearRegression() if len(df_feat) < 5 else GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    last_score = df["score"].iloc[-1]
    prev_score = df["score"].iloc[-2] if len(df) >= 2 else last_score
    rolling3 = df["score"].tail(3).mean()
    trend = last_score - prev_score
    predicted = model.predict(np.array([[last_score, prev_score, rolling3, trend]]))[0]
    return round(max(0.0, min(float(predicted), 100.0)), 2)


# -------------------- STREAK --------------------
def get_user_average(user_id, days=7):
    try:
        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT score FROM activity WHERE user_id=%s ORDER BY date DESC LIMIT %s",
            conn, params=(user_id, days))
        conn.close()
        return round(float(df["score"].mean()), 2) if not df.empty else None
    except Exception:
        return None


def update_streak(user_id, today_score, target, today_date):
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_streak, longest_streak, last_date, target FROM streak WHERE user_id=%s",
            (user_id,))
        row = cursor.fetchone()

        if row is None:
            hit = 1 if today_score >= target else 0
            longest = hit
            cursor.execute(
                "INSERT INTO streak (user_id, target, current_streak, longest_streak, last_date) VALUES (%s,%s,%s,%s,%s)",
                (user_id, target, hit, longest, today_date))
            conn.commit()
            conn.close()
            return {"current_streak": hit, "longest_streak": longest,
                    "target": target, "hit": bool(hit), "already_done_today": False}

        current, longest, last_date, saved_target = row
        use_target = target if target else saved_target

        # Same day — don't update again
        if last_date == today_date:
            conn.close()
            return {"current_streak": current, "longest_streak": longest,
                    "target": use_target, "hit": today_score >= use_target,
                    "already_done_today": True}

        # New day
        if today_score >= use_target:
            current += 1
            hit = True
        else:
            current = 0
            hit = False

        longest = max(longest, current)
        cursor.execute(
            "UPDATE streak SET current_streak=%s, longest_streak=%s, last_date=%s, target=%s WHERE user_id=%s",
            (current, longest, today_date, use_target, user_id))
        conn.commit()
        conn.close()
        return {"current_streak": current, "longest_streak": longest,
                "target": use_target, "hit": hit, "already_done_today": False}
    except Exception as e:
        return {"current_streak": 0, "longest_streak": 0,
                "target": target, "hit": False, "error": str(e)}


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


def generate_plot(user_id, conn):
    """Reuses existing DB connection — no new connection needed."""
    df = pd.read_sql_query(
        "SELECT date, score, screen_time, sleep, study, stress FROM activity WHERE user_id=%s",
        conn, params=(user_id,))

    if df.empty:
        return None, "Not enough data", "Keep going!"

    if len(df) >= 2:
        trend = ("Improving 📈" if df.iloc[-1]["score"] > df.iloc[-2]["score"]
                 else "Declining 📉" if df.iloc[-1]["score"] < df.iloc[-2]["score"]
        else "Same 😐")
    else:
        trend = "Not enough data"

    latest = df.iloc[-1]
    advice = []
    if latest["screen_time"] > 5: advice.append("Reduce screen time 📱")
    if latest["sleep"] < 6: advice.append("Improve sleep 😴")
    if latest["study"] < 2: advice.append("Study more 📚")
    if latest["stress"] > 7: advice.append("Try to relax 🧘")
    if not advice:                advice.append("Good job! Keep it up 👍")
    advice_text = " | ".join(advice)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
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
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
                       (username.lower(), password))
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
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        session["user_id"] = user[0]
        return jsonify({"message": "Login success", "user_id": user[0], "username": username})
    return jsonify({"message": "Login failed"})


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/reset_db")
def reset_db():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS activity")
    cursor.execute("DROP TABLE IF EXISTS friends")
    cursor.execute("DROP TABLE IF EXISTS streak")
    conn.commit()
    conn.close()
    init_db()
    return "DB Reset Done"


# -------------------- STREAK ROUTES --------------------
@app.route("/get_streak", methods=["POST"])
def get_streak():
    user_id = request.get_json(force=True).get("user_id")
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT current_streak, longest_streak, target FROM streak WHERE user_id=%s", (user_id,))
        row = cursor.fetchone()
        avg = get_user_average(user_id, days=7)
        conn.close()
        if row is None:
            return jsonify({"current_streak": 0, "longest_streak": 0, "target": None, "avg": avg})
        return jsonify({"current_streak": row[0], "longest_streak": row[1], "target": row[2], "avg": avg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set_target", methods=["POST"])
def set_target():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    target = float(data.get("target", 0))
    avg = get_user_average(user_id, days=7)
    if avg is not None and target <= avg:
        return jsonify(
            {"error": f"Target must be greater than your 7-day average ({avg}). Try {round(avg + 1, 1)}+"}), 400
    if target < 0 or target > 100:
        return jsonify({"error": "Target must be between 0 and 100"}), 400
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM streak WHERE user_id=%s", (user_id,))
        exists = cursor.fetchone()
        if exists:
            cursor.execute(
                "UPDATE streak SET target=%s, current_streak=0, last_date=NULL WHERE user_id=%s",
                (target, user_id))
        else:
            cursor.execute(
                "INSERT INTO streak (user_id, target, current_streak, longest_streak, last_date) VALUES (%s,%s,0,0,NULL)",
                (user_id, target))
        conn.commit()
        conn.close()
        return jsonify({"message": "Target set!", "target": target, "avg": avg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- FRIENDS --------------------
@app.route("/add_friend", methods=["POST"])
def add_friend():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    friend_username = data.get("friend_username", "").lower().strip()
    if not friend_username:
        return jsonify({"error": "Username required"}), 400
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=%s", (friend_username,))
    friend = cursor.fetchone()
    if not friend:
        conn.close()
        return jsonify({"message": "User not found"})
    friend_id = friend[0]
    if friend_id == user_id:
        conn.close()
        return jsonify({"error": "Cannot add yourself"})
    cursor.execute("SELECT 1 FROM friends WHERE user_id=%s AND friend_id=%s", (user_id, friend_id))
    if cursor.fetchone():
        conn.close()
        return jsonify({"error": "Already friends"})
    cursor.execute("INSERT INTO friends (user_id, friend_id) VALUES (%s, %s)", (user_id, friend_id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Friend added"})


@app.route("/get_friends", methods=["POST"])
def get_friends():
    user_id = request.get_json(force=True)["user_id"]
    conn = get_conn()
    df = pd.read_sql_query("""
                           SELECT u.id AS friend_id, u.username, MAX(a.score) AS best_score
                           FROM friends f
                                    JOIN users u ON u.id = f.friend_id
                                    LEFT JOIN activity a ON a.user_id = f.friend_id
                           WHERE f.user_id = %s
                           GROUP BY u.id, u.username
                           ORDER BY u.username
                                """, conn, params=(user_id,))
    conn.close()
    return jsonify({"friends": df.to_dict(orient="records")})


@app.route("/remove_friend", methods=["POST"])
def remove_friend():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    friend_id = data.get("friend_id")
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM friends WHERE user_id=%s AND friend_id=%s", (user_id, friend_id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Friend removed"})


# -------------------- LEADERBOARD --------------------
@app.route("/leaderboard", methods=["POST"])
def leaderboard():
    user_id = request.get_json(force=True)["user_id"]
    conn = get_conn()
    cursor = conn.cursor()

    df_friends = pd.read_sql_query("""
                                   SELECT u.username, MAX(a.score) AS score
                                   FROM activity a
                                            JOIN users u ON u.id = a.user_id
                                   WHERE a.user_id IN (SELECT friend_id FROM friends WHERE user_id = %s)
                                   GROUP BY u.username
                                   """, conn, params=(user_id,))

    df_me = pd.read_sql_query("""
                              SELECT u.username, MAX(a.score) AS score
                              FROM activity a
                                       JOIN users u ON u.id = a.user_id
                              WHERE a.user_id = %s
                              GROUP BY u.username
                              """, conn, params=(user_id,))

    cursor.execute("SELECT username FROM users WHERE id=%s", (user_id,))
    row = cursor.fetchone()
    my_username = row[0] if row else ""
    conn.close()

    df_combined = pd.concat([df_friends, df_me], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset="username")
    df_combined = df_combined.sort_values("score", ascending=False).reset_index(drop=True)
    result = df_combined.to_dict(orient="records")
    for r in result:
        r["is_me"] = (r["username"] == my_username)
    return jsonify(result)


# -------------------- ANALYZE --------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    load_model()

    user_id = data.get("user_id")
    try:
        user_id = int(user_id)
    except Exception:
        return jsonify({"error": "Invalid user_id"}), 400

    screen = float(data.get("screen_time") or 0)
    sleep = float(data.get("sleep") or 0)
    study = float(data.get("study") or 0)
    stress = float(data.get("stress") or 0)

    is_valid, error_msg = validate_inputs(screen, sleep, study, stress)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    text = data.get("text") or ""
    pred, prob = get_personality_prediction(text)
    score = calculate_score(screen, sleep, study, stress)

    ist = pytz.timezone("Asia/Kolkata")
    date = datetime.now(ist).strftime("%Y-%m-%d")

    # ── Single DB connection for insert + select + plot ──
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO activity (user_id, screen_time, sleep, study, stress, score, date) VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (user_id, screen, sleep, study, stress, score, date))
    conn.commit()

    df = pd.read_sql_query(
        "SELECT date, score FROM activity WHERE user_id=%s ORDER BY date",
        conn, params=(user_id,))

    graph, trend, advice = generate_plot(user_id, conn)  # reuse conn
    conn.close()

    # Retrain in background — doesn't block response
    threading.Thread(target=retrain_rf_with_db, daemon=True).start()

    progress = 0
    if len(df) >= 2:
        progress = score - df.iloc[-2]["score"]

    predicted = predict_score(df)

    # Streak
    streak_target = None
    try:
        conn2 = get_conn()
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT target FROM streak WHERE user_id=%s", (user_id,))
        t = cursor2.fetchone()
        conn2.close()
        if t and t[0]:
            streak_target = float(t[0])
    except Exception:
        pass

    streak_info = {}
    if streak_target is not None:
        streak_info = update_streak(user_id, score, streak_target, date)

    avg = get_user_average(user_id, days=7)

    return jsonify({
        "personality": pred,
        "confidence": round(prob * 100, 2),
        "score": score,
        "progress": progress,
        "suggestion": give_suggestion(pred, score, progress),
        "graph": f"data:image/png;base64,{graph}" if graph else None,
        "trend": trend,
        "advice": advice,
        "prediction": predicted,
        "streak": streak_info,
        "avg": avg
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