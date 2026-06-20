# 📊 Advanced Text Analytics & Sentiment Pipeline Dashboard

An interactive Natural Language Processing (NLP) web application built using **Gradio 6.0** and **FastAPI**. The system systematically cleans raw text, extracts structural features via a TF-IDF vectorizer mapping against a 25,000-word vocabulary, and classifies text using an optimized Linear Support Vector Machine (LinearSVC) classifier boundary.

The architecture features a **centralized control engine** (`interactive_test.py`) that acts as a unified hub for local staging, testing, and UI deployment.

---

## ⚙️ Core Processing Pipeline Architecture

When a text phrase is entered, it passes through three sequential architectural layers:

1. **Stage 1: Text Sanitization (Regex Layer)**
   Normalizes the text shape by forcing lowercasing, stripping punctuation, deleting numeric noise, and removing symbols or emojis.
2. **Stage 2: Sparse Matrix Tokenization (TF-IDF Vectorizer)**
   Maps the sanitized string tokens directly against the model's high-dimensional feature coordinates, calculating individual word significance weights.
3. **Stage 3: Classifier Boundary (LinearSVC Machine Learning Model)**
   Feeds the sparse coordinates into the trained SVM model to instantly determine whether the phrase lands in a positive or negative semantic zone.

---

## 📁 Repository Structure

```text
├── 📄 app.py                # Visual Gradio Dashboard UI with FastAPI wrapper
├── 📄 interactive_test.py   # Centralized system control engine (The Hub)
├── 📄 train_model.py        # Machine learning training & feature serialization pipeline
├── 📄 requirements.txt      # Locked environment dependency manifest
├── 📄 vercel.json           # Cloud deployment & routing rules configuration
├── 📄 .gitignore            # Excludes environment folders (venv) and local caches
├── 📦 sentiment_model.pkl   # Serialized LinearSVC model weights checkpoint
└── 📦 vectorizer.pkl        # Serialized TF-IDF vectorizer vocabulary coordinates
