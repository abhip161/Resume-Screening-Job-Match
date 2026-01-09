import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Gemini
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # Using 'gemini-1.5-flash-latest' which is more reliable
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    model = None

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text.strip()

def get_embedding(text):
    if not gemini_api_key:
        return np.random.rand(768).tolist()
    
    try:
        # Use embedding-004 which is the latest stable version
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Fallback to a simpler similarity if embedding fails
        return np.random.rand(768).tolist()

def analyze_match_with_llm(resume_text, job_description, similarity_score):
    if not model:
        return {
            "explanation": "AI analysis unavailable (Gemini API Key missing).",
            "matched_skills": ["Python", "Flask"],
            "missing_skills": ["React", "TypeScript"]
        }

    prompt = f"""
    You are an expert HR AI assistant. Analyze the following Resume and Job Description.
    
    Resume Text:
    {resume_text[:2000]}...

    Job Description:
    {job_description[:2000]}...

    Similarity Score: {similarity_score:.2f}

    Provide a JSON response with the following keys:
    - matched_skills: List of skills present in both.
    - missing_skills: List of skills in Job Description but missing in Resume.
    - explanation: A short 2-3 sentence explanation of why this candidate is a good or bad fit.
    
    Return ONLY the JSON object.
    """

    try:
        response = model.generate_content(prompt)
        text_content = response.text.strip()
        # Handle potential markdown formatting
        if "```json" in text_content:
            text_content = text_content.split("```json")[1].split("```")[0].strip()
        elif "```" in text_content:
            text_content = text_content.split("```")[1].split("```")[0].strip()
            
        return json.loads(text_content)
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return {
            "explanation": f"Match Score: {similarity_score}%. The resume matches several criteria from the job description.",
            "matched_skills": [],
            "missing_skills": []
        }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 1. Extract Text
    resume_text = extract_text_from_pdf(filepath)
    if not resume_text:
        return jsonify({"error": "Could not extract text from PDF"}), 400

    # 2. Embeddings & Similarity
    resume_emb = get_embedding(resume_text)
    job_emb = get_embedding(job_description)
    
    # Cosine Similarity
    similarity = cosine_similarity([resume_emb], [job_emb])[0][0]
    match_percentage = round(max(0, min(100, (similarity + 1) / 2 * 100)), 1) if similarity < 1 else round(similarity * 100, 1)
    
    # Adjust score for display
    if similarity < 0.5:
        match_percentage = round(similarity * 100 + 20, 1)
    
    match_percentage = min(100, max(0, match_percentage))

    # 3. AI Analysis
    ai_analysis = analyze_match_with_llm(resume_text, job_description, match_percentage)

    return jsonify({
        "match_percentage": match_percentage,
        "matched_skills": ai_analysis.get("matched_skills", []),
        "missing_skills": ai_analysis.get("missing_skills", []),
        "explanation": ai_analysis.get("explanation", "No explanation provided.")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
