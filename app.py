import streamlit as st
from summarize import LegalSummarizer
import time
import textwrap

# -----------------------------------------------------------
# ⚖️ STREAMLIT APP CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="⚖️ Legal Case Summarizer",
    layout="wide",
    page_icon="⚖️"
)

st.title("⚖️ Legal Case Summarizer ")


# -----------------------------------------------------------
# 🔧 LOAD MODELS (cached for performance)
# -----------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    return LegalSummarizer()

summarizer = load_models()

# -----------------------------------------------------------
# 🧾 INPUT SECTION
# -----------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    uploaded = st.file_uploader("Upload a court judgment (PDF)", type=["pdf"])
    detail = st.selectbox(
        "Summary length:",
        ["Short (digest)", "Detailed (1-3 pages)"],
        index=1
    )
    btn = st.button("Generate Summary")

with col2:
    st.markdown("### ⚙️ Tips")
    st.markdown("- Use **Detailed** for full legal analysis; **Short** for quick reading.")
    st.markdown("- You can download the structured summary once generated.")
    

# -----------------------------------------------------------
# 🧠 SUMMARIZATION LOGIC
# -----------------------------------------------------------
if uploaded:
    bytes_pdf = uploaded.read()
    st.info("✅ PDF uploaded successfully.")

    if btn:
        with st.spinner("Summarizing... this may take a few moments for long judgments"):
            start = time.time()

            try:
                detail_map = {
                    "Short (digest)": "short",
                    "Detailed (1-3 pages)": "detailed"
                }
                mapped_detail = detail_map.get(detail, "detailed")

                # Always run the hybrid summarizer (no extractive option exposed)
                out = summarizer.summarize_hybrid(bytes_pdf, detail=mapped_detail)

                elapsed = time.time() - start
                st.success(f"✅ Summary generated in {elapsed:.1f} seconds.")

                # -----------------------------------------------------------
                # 🧾 SIDEBAR METADATA EXTRACTION
                # -----------------------------------------------------------
                if out.startswith("## ⚖️ Case Metadata"):
                    parts = out.split("\n\n", 2)
                    meta_block = parts[0]
                    rest = parts[2] if len(parts) > 2 else ""
                else:
                    meta_block, rest = "", out

                if meta_block:
                    st.sidebar.header("⚖️ Case Metadata")
                    for line in meta_block.splitlines()[1:]:
                        if line.strip():
                            keyval = line.lstrip("- ").split(":", 1)
                            if len(keyval) == 2:
                                k, v = keyval
                                st.sidebar.markdown(f"**{k.strip()}**: {v.strip()}")

                # -----------------------------------------------------------
                # 📘 MAIN SUMMARY OUTPUT
                # -----------------------------------------------------------
                st.markdown(rest)

                # Plain text download option
                with st.expander("📄 View Plain Text / Download"):
                    st.text_area("Summary (plain text)", textwrap.fill(out, width=110), height=400)
                    st.download_button(
                        "💾 Download Summary (.txt)",
                        out.encode("utf-8"),
                        file_name="legal_summary.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"❌ Summarization failed: {e}")

else:
    st.info("📂 Please upload a PDF to begin.")
