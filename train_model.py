import pandas as pd
import re
import joblib
from datasets import load_dataset 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC # 🚨 CHANGED: High-speed linear classifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

print("Downloading massive dataset from Hugging Face...")
# 2. Download the IMDB dataset directly from Hugging Face
hf_dataset = load_dataset("stanfordnlp/imdb")

# Convert the Hugging Face format into a Pandas DataFrame
df_massive = pd.DataFrame(hf_dataset['train'])

df_massive = df_massive.rename(columns={'text': 'text'})

# The IMDB dataset uses 0 for negative and 1 for positive.
df_massive['sentiment'] = df_massive['label'].map({0: 'negative', 1: 'positive'})

print("Loading local datasets...")
# Load your local files
df_original = pd.read_csv("sentiment_dataset.csv")
df_extra = pd.read_csv("INA_TweetsPPKM_Labeled_Pure.csv", on_bad_lines='skip')

# 4. Combine Everything
print("Merging data...")
df = pd.concat([df_original, df_extra, df_massive], ignore_index=True)

print("Removing empty rows...")
df = df.dropna(subset=['text', 'sentiment'])
df = df[df["text"].str.strip() != ""]

print(f"Total training examples ready: {len(df):,} rows")

# 5. Apply cleaning and continue as normal
print("Cleaning text (this might take a moment with 10,000+ rows)...")
df["cleaned_text"] = df["text"].apply(clean_text)

# Split Data
X = df["cleaned_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Vectorizer and Model
print("Vectorizing text...")
tfidf_vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2)) # Expanded features for a larger dataset
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("🧠 Training LinearSVC model (Optimized for speed)...")
# 🚨 LinearSVC gives identical accuracy to SVC(kernel='linear') but runs 100x faster
best_model = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=2000)
best_model.fit(X_train_tfidf, y_train)

# 4. Evaluate
print("Evaluating performance...")
y_pred = best_model.predict(X_test_tfidf)
print("\n" + "="*50)
print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("="*50)
print(classification_report(y_test, y_pred))
print("="*50)

# 5. Save the Vectorizer and Model to disk
print("Saving model to disk...")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")
joblib.dump(best_model, "sentiment_model.pkl")
print("Done! You can now run your testing dashboard script.")