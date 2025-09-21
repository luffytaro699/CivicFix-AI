import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------- 1. Load data ----------
df = pd.read_csv("ai_services/dataset/text_dataset/balanced_varied_text_dataset.csv")

# ---------- 2. Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    return text

X = df["complaint"].apply(clean_text)
y = df["department"]

# ---------- 3. TF-IDF Vectorization ----------
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=2000,     # increased vocabulary
    ngram_range=(1,2)      # include bigrams
)

X_vec = vectorizer.fit_transform(X)

# ---------- 4. Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. Train model ----------
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# ---------- 6. Evaluate ----------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------- 7. Save model and vectorizer ----------
joblib.dump(model, "ai_services/models/dept_model.pkl")
joblib.dump(vectorizer, "ai_services/models/vectorizer.pkl")

print("Model and vectorizer saved successfully ✅")
