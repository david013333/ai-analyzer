from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Step 1 — Loading SentenceTransformer model...")
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("         Done!")

print("Step 2 — Loading dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
texts = df["text"].tolist()
labels = df["label"].tolist()
print(f"         {len(texts)} rows loaded!")

print("Step 3 — Encoding all sentences...")
X = st.encode(texts, show_progress_bar=True)
print("         Done!")

print("Step 4 — Training Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    C=5.0,
    class_weight="balanced",
    solver="lbfgs"
)
model.fit(X, labels)
print("         Done!")

print("Step 5 — Saving pkl files...")
joblib.dump(st, os.path.join(BASE_DIR, "st_model.pkl"))
joblib.dump(model, os.path.join(BASE_DIR, "personality_model.pkl"))

# Show file sizes
sz_st = os.path.getsize(os.path.join(BASE_DIR, "st_model.pkl")) / (1024 * 1024)
sz_model = os.path.getsize(os.path.join(BASE_DIR, "personality_model.pkl")) / (1024 * 1024)

print()
print("=" * 50)
print(f"st_model.pkl          → {sz_st:.1f} MB")
print(f"personality_model.pkl → {sz_model:.2f} MB")
print()
print("Now do:")
print("  git add st_model.pkl personality_model.pkl")
print("  git commit -m 'Added pretrained pkl files'")
print("  git push")
print("=" * 50)