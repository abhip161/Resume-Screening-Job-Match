# Resume Screening & Job Match Application

A Flask-based web application that compares a candidate’s resume with a job description and provides:
- Overall match percentage
- Matched skills
- Missing skills
- A concise explanation of alignment

The system uses **local transformer embeddings** for semantic similarity and **rule-based logic** for skill gap analysis, without relying on any external or paid AI APIs.

---

## 🚀 Features

- Upload resume in PDF format
- Paste job description text
- Semantic similarity scoring using transformer embeddings
- Skill matching and gap identification
- Simple, clean UI
- Fully local processing (no API keys required)

---

## 🧠 How It Works

1. Resume text is extracted from the uploaded PDF.
2. Resume and job description are converted into vector embeddings using a local transformer model.
3. Cosine similarity is used to compute an overall match percentage.
4. Skills are extracted using a predefined rule-based skill list.
5. Results are displayed with a short, human-readable explanation.

> Note: The match score represents **semantic similarity**, while missing skills are shown separately to highlight gaps.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **ML / NLP:** sentence-transformers (all-MiniLM-L6-v2)  
- **Similarity:** scikit-learn (cosine similarity)  
- **PDF Processing:** PyPDF2  
- **Frontend:** HTML, Bootstrap  

---

## 📂 Project Structure

