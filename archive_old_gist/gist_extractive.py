import sys
import types
from sklearn.preprocessing import LabelEncoder

# 🧩 Compatibility patch for older sklearn models
if 'sklearn.preprocessing.label' not in sys.modules:
    sklearn_preproc = types.ModuleType("sklearn.preprocessing.label")
    sklearn_preproc.LabelEncoder = LabelEncoder
    sys.modules['sklearn.preprocessing.label'] = sklearn_preproc

import nltk
import numpy as np
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import joblib
import re


class GistExtractiveSummarizer:
    """
    Extractive legal summarizer based on the Gist model.
    Combines pretrained embeddings (300D) + handcrafted legal features (34D).
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.summary_model_path = os.path.join(model_dir, "summary_model.pkl")
        self.word2vec_path = os.path.join(model_dir, "word2vec_model.bin")

        # Step 1: Load classifier
        print("🔹 Loading summary model (scikit-learn LightGBM classifier)...")
        try:
            self.summary_model = joblib.load(self.summary_model_path)
            print("✅ scikit-learn LightGBM model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load summary model: {e}")

        # Step 2: Load embeddings
        print("🔹 Loading word2vec embeddings (trying multiple formats)...")
        try:
            self.word2vec = KeyedVectors.load_word2vec_format(
                self.word2vec_path, binary=True
            )
            print("✅ Loaded embeddings via load_word2vec_format()")
        except Exception as e1:
            print(f"⚠️ Binary format failed: {e1}")
            print("🔁 Trying gensim Word2Vec.load() instead...")
            try:
                self.word2vec = gensim.models.Word2Vec.load(self.word2vec_path)
                if hasattr(self.word2vec, "wv"):
                    self.word2vec = self.word2vec.wv
                    print("🔁 Extracted KeyedVectors from Word2Vec model.")
            except Exception as e2:
                raise RuntimeError(f"❌ Failed to load word2vec model: {e2}")

        self.stop_words = set(stopwords.words("english"))

    # ---------------------------------------------------------------
    # ✅ Utility: Create consistent sentence vectors (300D)
    # ---------------------------------------------------------------
    def sentence_vector(self, sentence):
        words = [
            w.lower()
            for w in word_tokenize(sentence)
            if w.isalpha() and w.lower() not in self.stop_words
        ]
        if not words:
            return np.zeros(300, dtype=np.float32)

        valid_vectors = []
        for w in words:
            if w in self.word2vec:
                vec = np.array(self.word2vec[w], dtype=np.float32).flatten()
                if vec.shape[0] == 300:
                    valid_vectors.append(vec)

        if len(valid_vectors) == 0:
            return np.zeros(300, dtype=np.float32)

        return np.mean(valid_vectors, axis=0)

    # ---------------------------------------------------------------
    # ✅ Utility: Generate handcrafted features (34D)
    # ---------------------------------------------------------------
    def legal_features(self, sentence, position, total_sentences):
        """
        Simple handcrafted features mimicking the original Gist features.
        """
        length = len(sentence)
        word_count = len(sentence.split())
        avg_word_len = np.mean([len(w) for w in sentence.split()]) if word_count else 0
        numeric_count = len(re.findall(r'\d+', sentence))
        has_section = 1 if re.search(r'\b(section|article|act|u/s|ipc)\b', sentence, re.I) else 0
        position_norm = position / max(total_sentences, 1)

        # Basic handcrafted feature vector
        features = np.array([
            length / 1000, word_count / 100, avg_word_len / 10,
            numeric_count / 10, has_section, position_norm
        ])

        # Pad to reach 34 dimensions (as model expects)
        if len(features) < 34:
            features = np.pad(features, (0, 34 - len(features)))

        return features.astype(np.float32)

    # ---------------------------------------------------------------
    # ✅ Main summarization
    # ---------------------------------------------------------------
    def summarize(self, text, top_n=10):
        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return " ".join(sentences)

        # Build combined features (300D + 34D)
        combined_features = []
        total_sent = len(sentences)
        for i, s in enumerate(sentences):
            emb = self.sentence_vector(s)
            extra = self.legal_features(s, i, total_sent)
            combined = np.concatenate([emb, extra])  # 334D total
            combined_features.append(combined)

        X = np.array(combined_features)

        # Predict importance
        y_pred = self.summary_model.predict_proba(X)[:, 1]

        # Rank and select top sentences
        top_idx = np.argsort(y_pred)[-top_n:][::-1]
        selected = [sentences[i] for i in top_idx]

        return " ".join(selected)
