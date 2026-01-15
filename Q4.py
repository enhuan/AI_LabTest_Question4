import streamlit as st
import nltk
from PyPDF2 import PdfReader

# Ensure punkt is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Streamlit page setup
st.set_page_config(page_title="PDF Sentence Chunker (NLTK)", layout="wide")
st.title("PDF Sentence Chunker Demo")
st.write(
    "Upload a PDF file, extract text, and split it into sentences using "
    "NLTK's `sent_tokenize`. Each sentence represents a semantic chunk."
)

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 2: Extract text
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        full_text = " ".join(pages_text).strip()

        st.subheader("Basic info")
        st.write(f"Number of pages: **{len(reader.pages)}**")
        st.write(f"Total characters extracted: **{len(full_text)}**")

        if not full_text:
            st.warning("No text could be extracted from this PDF.")
        else:
            # Step 3 & 4: Sentence tokenization (semantic chunking)
            sentences = nltk.sent_tokenize(full_text)
            st.success(f"Number of detected sentences: {len(sentences)}")

            # Display sentences 58 to 68
            start_idx = 58
            end_idx = 68
            sample_sentences = sentences[start_idx:end_idx] if len(sentences) > start_idx else sentences
            st.subheader(f"Sample sentences [{start_idx} : {end_idx})")
            for i, sent in enumerate(sample_sentences, start=start_idx):
                st.markdown(f"**{i}**. {sent}")

            # Show all sentences as semantic chunks
            with st.expander("Show all semantic sentence chunks"):
                for i, sent in enumerate(sentences):
                    st.markdown(f"**{i}**. {sent}")

            # Show first 2000 characters of raw text
            with st.expander("Show raw extracted text (first 2000 characters)"):
                st.text(full_text[:2000])

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
else:
    st.info("Please upload a PDF to begin.")
