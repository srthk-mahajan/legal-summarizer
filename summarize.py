"""
summarize.py
Enhanced LegalSummarizer (stable, CPU-friendly)

- Local Pegasus first (no internet required if model folder present)
- SBERT extractive (optional) with graceful fallback
- Safe, single abstractive_from_text() optimized for CPU
- Regex-only metadata extraction (no spaCy requirement)
- OCR-enabled (pdf2image + pytesseract) if installed
- Outputs well-structured markdown sections for Streamlit
"""

import os
import re
import io
import logging
import textwrap
from typing import Tuple, List, Dict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PyPDF2 import PdfReader

# optional Sentence-BERT
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    st_util = None
    _SBERT_AVAILABLE = False

# nltk sentence tokenizer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# optional OCR
OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    convert_from_bytes = None
    pytesseract = None
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalSummarizer")


# ---------------------------
# Fallback metadata extractor
# ---------------------------
def _fallback_extract_case_metadata(text: str) -> Dict[str, str]:
    out = {
        "Case Title": "N/A",
        "Court Name": "N/A",
        "Bench / Judges": "N/A",
        "Date of Judgment": "N/A",
        "Citation": "N/A",
        "Type of Case": "N/A",
    }

    # Case title pattern "X v. Y" or "X vs Y"
    m = re.search(r"([A-Z][A-Za-z\.\-&\s]{2,100}?)\s+v(?:\.|s\.|s)\s+([A-Z][A-Za-z\.\-&\s]{2,100}?)", text)
    if m:
        out["Case Title"] = m.group(0).strip()

    # Date of judgment (simple)
    d = re.search(
        r"\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b",
        text,
        flags=re.I,
    )
    if d:
        out["Date of Judgment"] = d.group(0)

    # Citation like (2012) 5 SCC 777 or AIR 1978 SC 597
    c = re.search(r"\(?\d{4}\)?\s*\d*\s*(SCC|AIR|CriLJ|SCR)\s*\d+", text, flags=re.I)
    if c:
        out["Citation"] = c.group(0)

    # Judges heuristics
    judges = re.findall(r"(Justice\s+[A-Z][A-Za-z\.\s,]+)", text)
    if judges:
        out["Bench / Judges"] = ", ".join(sorted(set(judges)))

    # Court name heuristics
    if re.search(r"Supreme Court of India", text, flags=re.I):
        out["Court Name"] = "Supreme Court of India"
    elif re.search(r"High Court", text, flags=re.I):
        out["Court Name"] = "High Court"

    # Type patterns
    case_type_patterns = [
        r"Criminal\s+Appeal",
        r"Civil\s+Appeal",
        r"Writ\s+Petition",
        r"Special\s+Leave\s+Petition",
        r"Civil\s+Suit",
        r"Criminal\s+Case",
        r"Appeal\s+No\.?",
        r"Petition\s+No\.?",
    ]
    for pat in case_type_patterns:
        mm = re.search(pat, text, flags=re.I)
        if mm:
            out["Type of Case"] = mm.group(0).strip()
            break

    # extra heuristic
    if out["Type of Case"] == "N/A":
        low = text.lower()
        if "criminal" in low:
            out["Type of Case"] = "Criminal"
        elif "writ" in low:
            out["Type of Case"] = "Writ Petition"
        elif "civil" in low:
            out["Type of Case"] = "Civil"

    return out


# ---------------------------
# Main summarizer class
# ---------------------------
class LegalSummarizer:
    def __init__(
        self,
        pegasus_local_dir: str = os.path.join("models", "legal-pegasus"),
        # Use a small, CDN-cached model that downloads reliably on Streamlit Cloud
        # (distilbart-cnn is lighter than pegasus and avoids LFS hassles)
        pegasus_fallback: str = "sshleifer/distilbart-cnn-12-6",
        sbert_model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        # device auto-detect (uses CPU if torch has no CUDA)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # SBERT optional
        self.sbert_available = False
        self.embedder = None
        if _SBERT_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer '{sbert_model_name}'")
                self.embedder = SentenceTransformer(sbert_model_name)
                self.sbert_available = True
            except Exception as e:
                logger.warning(f"SBERT load failed: {e}")
                self.embedder = None
                self.sbert_available = False
        else:
            logger.info("SBERT not installed — using heuristic extractive fallback.")

        # Pegasus local-first, fallback to hf
        if os.path.exists(pegasus_local_dir) and os.listdir(pegasus_local_dir):
            pegasus_model_dir = pegasus_local_dir
            logger.info(f"Using local Pegasus at {pegasus_local_dir}")
        else:
            pegasus_model_dir = pegasus_fallback
            logger.warning(f"Local Pegasus not found; falling back to '{pegasus_fallback}' (internet required)")

        try:
            logger.info(f"Loading summarization tokenizer/model from {pegasus_model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(pegasus_model_dir, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pegasus_model_dir).to(self.device)
            logger.info("Summarizer model ready.")
        except OSError as e:
            # Give a clear message on Streamlit Cloud if model download fails
            raise OSError(
                "Model download failed. Please redeploy or restart the app; "
                "ensure internet access to Hugging Face Hub."
            ) from e

    # ---------------------------
    # PDF extraction + OCR
    # ---------------------------
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, ocr_if_empty: bool = True, ocr_language: str = "eng") -> str:
        pages_text = []
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for p in reader.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = None
                if t:
                    pages_text.append(t)
        except Exception as e:
            logger.warning(f"PyPDF2 read failed: {e}")

        full = "\n".join(pages_text).strip()
        wc = len(full.split())
        logger.info(f"PyPDF2 extracted words: {wc}")

        # If very small, attempt OCR if available
        if (wc < 200) and ocr_if_empty and OCR_AVAILABLE:
            logger.info("Attempting OCR via pdf2image + pytesseract")
            try:
                images = convert_from_bytes(pdf_bytes, dpi=200)
                ocr_out = [pytesseract.image_to_string(img, lang=ocr_language) for img in images]
                ocr_full = "\n".join(ocr_out).strip()
                if len(ocr_full.split()) > wc:
                    full = ocr_full
            except Exception as e:
                logger.warning(f"OCR failed: {e}")

        return full

    # ---------------------------
    # Extract citations/statutes
    # ---------------------------
    def extract_citations_and_statutes(self, text: str) -> Tuple[List[str], List[str]]:
        statutes, cases = set(), set()
        for m in re.finditer(r"\b(?:Section|S\.|Sec\.|Article|Art\.)\s*\d{1,4}\b", text, flags=re.I):
            statutes.add(m.group(0))
        acts = [r"Indian Penal Code", r"IPC", r"CrPC", r"Evidence Act", r"Companies Act", r"Contract Act"]
        for a in acts:
            for m in re.finditer(r"\b" + re.escape(a) + r"\b", text, flags=re.I):
                statutes.add(m.group(0))
        for m in re.finditer(r"\b([A-Z][A-Za-z\.\-&\s]{2,80}?)\s+v(?:\.|s\.?|s)\s+([A-Z][A-Za-z\.\-&\s]{2,80}?)\b", text):
            cases.add(m.group(0))
        return sorted(list(statutes))[:50], sorted(list(cases))[:80]

    # ---------------------------
    # Extractive selection
    # ---------------------------
    def extract_representative_sentences(self, text: str, top_k: int = 16, max_words: int = 1200) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        if self.sbert_available and getattr(self, "embedder", None) is not None:
            try:
                embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
                doc_emb = embeddings.mean(dim=0, keepdim=True)
                cos_scores = st_util.cos_sim(embeddings, doc_emb).squeeze(1).cpu().numpy()
                ranked = sorted(range(len(sentences)), key=lambda i: float(cos_scores[i]), reverse=True)
                selected = []
                tw = 0
                for idx in ranked:
                    w = len(sentences[idx].split())
                    if len(selected) < top_k and (tw + w) <= max_words:
                        selected.append(idx)
                        tw += w
                    if len(selected) >= top_k or tw >= max_words:
                        break
                chosen = [sentences[i] for i in sorted(selected)]
                return " ".join(chosen)
            except Exception as e:
                logger.warning(f"SBERT extraction failed: {e}. Falling back to heuristic.")
        # fallback: first N sentences up to max_words
        chosen = []
        tw = 0
        for s in sentences:
            w = len(s.split())
            if (tw + w) > max_words:
                break
            chosen.append(s)
            tw += w
            if len(chosen) >= top_k:
                break
        return " ".join(chosen)

    # ---------------------------
    # Abstractive (Pegasus) - CPU-safe single function
    # ---------------------------
    def abstractive_from_text(self, input_text: str, max_length: int = 850, min_length: int = 300) -> str:
        """
        Safe, CPU-friendly Pegasus summarizer.
        Produces clearly sectioned output and avoids truncation artifacts.
        """
        input_text = re.sub(r"\s+", " ", input_text).strip()
        if len(input_text.split()) < 60:
            return input_text

        # Hard context cap (~2800 words) — keeps CPU Pegasus stable
        words = input_text.split()
        snippet = " ".join(words[:2800])

        # Short, strong structured prompt
        prompt = f"""
Summarize the following Indian court judgment into a formal, well-structured case note.
Use concise legal language and clear markdown sections as below:

## ⚖️ Case Metadata
- Case Title:
- Court Name:
- Bench / Judge(s):
- Date of Judgment:
- Citation:
- Type of Case:

## 🧾 Background / Facts
(Brief factual and procedural background.)

## ❓ Issues for Determination
(Clearly list the legal questions before the court.)

## 🗣️ Arguments by Both Sides
(Key submissions of petitioner/appellant and respondent.)

## ⚖️ Court’s Reasoning / Analysis
(Explain the court’s reasoning and interpretation of law.)

## 📜 Decision / Holding
(Outcome, orders, or relief granted.)

## 💡 Key Takeaways / Legal Principles
(Short bullets summarizing the principles laid down.)

Judgment text:
{snippet}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1280).to(self.device)

        try:
            gen = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=2,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            summary = self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()

            # Cleanup & formatting
            summary = re.sub(r"\n{2,}", "\n\n", summary)
            summary = re.sub(r"(\.)([A-Z])", r". \2", summary)
            # if not summary.startswith("## ⚖️"):
            #     summary = "## 📘 Structured Headnote Summary\n\n" + summary
            summary = re.sub(r"\s{2,}", " ", summary)
            summary = re.sub(r"(\[\[|\]\]|\([0-9]+\))", "", summary)
            return summary

        except Exception as e:
            logger.warning(f"Pegasus generation failed: {e}. Returning extract snippet.")
            # Return a readable extract snippet as fallback (not empty)
            return snippet

    # ---------------------------
    # Hybrid pipeline: extractive -> abstractive
    # ---------------------------
    def summarize_hybrid(self, pdf_bytes: bytes, detail: str = "short") -> str:
        text = self.extract_text_from_pdf_bytes(pdf_bytes)
        if not text or len(text.split()) < 20:
            return "Error: Could not extract usable text from the uploaded PDF."

        # choose extraction + generation sizes by detail
        if detail == "short":
            top_k, max_len, min_len = 8, 400, 150
        else:  # 'detailed'
            top_k, max_len, min_len = 22, 1100, 450

        extract = self.extract_representative_sentences(text, top_k=top_k, max_words=1400)
        if not extract or len(extract.split()) < 60:
            extract = " ".join(text.split()[:1600])

        try:
            abstr = self.abstractive_from_text(extract, max_length=max_len, min_length=min_len)
        except Exception as e:
            logger.warning(f"Pegasus generation failed: {e}. Using extract as fallback.")
            abstr = extract

        statutes, cases = self.extract_citations_and_statutes(text)
        meta = _fallback_extract_case_metadata(text)

        # Build metadata markdown block
        meta_lines = ["## ⚖️ Case Metadata"]
        for k, v in meta.items():
            emoji = "🧾" if k == "Case Title" else "🏛️" if k == "Court Name" else "⚖️" if "Judge" in k else "📅" if "Date" in k else "📘" if "Citation" in k else "📂"
            meta_lines.append(f"{emoji} **{k}**: {v}")
        meta_block = "\n".join(meta_lines)

        # Assemble final output
        output = f"{meta_block}\n\n---\n\n## 📘 Structured Headnote Summary\n\n{abstr}\n\n"
        if statutes or cases:
            output += "## 📚 Citations & Statutes Detected\n"
            if statutes:
                output += f"- **Statutes/Sections:** {', '.join(statutes)}\n"
            if cases:
                output += f"- **Case Citations (Heuristic):** {', '.join(cases)}\n"
        return output

    # ---------------------------
    # Extractive-only pipeline
    # ---------------------------
    def summarize_extractive_from_pdf(self, pdf_bytes: bytes, top_k: int = 12) -> str:
        text = self.extract_text_from_pdf_bytes(pdf_bytes)
        return self.extract_representative_sentences(text, top_k=top_k)


# ---------------------------
# CLI test
# ---------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python summarize.py <path-to-pdf>")
        exit(0)
    path = sys.argv[1]
    with open(path, "rb") as f:
        b = f.read()
    s = LegalSummarizer()
    print(s.summarize_hybrid(b, detail="detailed"))
