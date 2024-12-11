import streamlit as st
import fitz  # PyMuPDF for PDF processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with fitz.open(pdf_file) as pdf:
        for page in pdf:
            pdf_text += page.get_text()
    return pdf_text

# Function to find the most relevant PDFs
def find_matching_pdfs(job_description, pdf_files):
    pdf_texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
    vectorizer = TfidfVectorizer()
    documents = [job_description] + pdf_texts
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    results = sorted(zip(pdf_files, similarity_scores), key=lambda x: x[1], reverse=True)
    return results

# Streamlit app
st.title("PDF Matcher for Job Descriptions")
st.write("Upload PDFs and a job description to find the best match!")

# Input: Job description
job_desc_input = st.text_area("Enter the Job Description", height=150)

# Input: PDF files
uploaded_pdfs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

# Button to process
if st.button("Find Matching PDFs"):
    if job_desc_input and uploaded_pdfs:
        # Process files
        matches = find_matching_pdfs(job_desc_input, uploaded_pdfs)
        st.subheader("Results:")
        for file, score in matches:
            st.write(f"File: {file.name}, Relevance Score: {score:.2f}")
    else:
        st.error("Please upload PDFs and enter a job description!")
