import io
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# System initialization & data downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Enterprise AI Career Engine")

# Configure CORS so our frontend can safely communicate across ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Transformer weights into memory once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))

# Mock Database Store for Curated Job Openings (Full-Time & Internships)
JOBS_DATABASE = [
    {
        "id": 1,
        "title": "Junior Python Developer / Data Analyst",
        "company": "TechNova Solutions",
        "type": "Internship",
        "location": "Remote",
        "link": "https://example.com/apply/technova-python",
        "description": "We are seeking an intern proficient in Python programming. Hands-on experience cleaning and parsing unstructured datasets using Pandas and NumPy arrays is required. Basic understanding of training predictive models using Scikit-Learn is highly preferred. Candidate must write optimized SQL queries."
    },
    {
        "id": 2,
        "title": "Cloud DevOps & Infrastructure Engineer",
        "company": "Apex Cloud Systems",
        "type": "Full-Time",
        "location": "Hybrid (New York)",
        "link": "https://example.com/apply/apex-devops",
        "description": "Looking for a full-time DevOps engineer specializing in infrastructure automation. Core responsibilities include managing production workloads inside Kubernetes clusters and writing custom Docker files. Building deployment automation pipelines using Jenkins CI/CD pipelines, provisioning cloud environments via Terraform IaC configurations, and handling AWS administrative operations."
    },
    {
        "id": 3,
        "title": "Frontend Software Engineer (React)",
        "company": "PixelPerfect UI Labs",
        "type": "Full-Time",
        "location": "On-Site",
        "link": "https://example.com/apply/pixelperfect-react",
        "description": "Join our product team to build high-performance user interfaces. Must have excellent mastery of React.js components, state management workflows, and building modern designs using Tailwind CSS configurations. Experience integrating backend microservices via REST API Fetch interfaces, handling JSON data streams, and using Git version control systems is mandatory."
    }
]

def preprocess_text(text: str) -> str:
    """Cleans raw text data and isolates valuable tokens."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', '', text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words])

@app.get("/api/jobs")
async def get_jobs():
    """Fetches list of active target vacancies."""
    return JOBS_DATABASE

@app.post("/api/analyze")
async def analyze_resume(
    job_id: int = Form(...),
    file: UploadFile = File(...)
):
    # Find the corresponding job entry from our store
    job_record = next((j for j in JOBS_DATABASE if j["id"] == job_id), None)
    if not job_record:
        raise HTTPException(status_code=404, detail="The targeted job profile record was not found.")

    # Parse uploaded file bytes
    try:
        pdf_bytes = await file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        resume_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text + "\n"
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to cleanly compile text from the uploaded PDF document.")

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="The provided PDF document has no extractable text characters.")

    # Run SBERT contextual similarity calculations
    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(job_record["description"])

    resume_embedding = model.encode(clean_resume, convert_to_tensor=True)
    jd_embedding = model.encode(clean_jd, convert_to_tensor=True)
    
    similarity = util.cos_sim(resume_embedding, jd_embedding).item()
    score = max(0.0, min(100.0, round(similarity * 100, 1)))

    # Process keyword extraction via TF-IDF if match score falls under the 50% baseline rule
    missing_keywords = []
    ai_suggestions = []

    if score < 50.0:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=120)
        tfidf_matrix = vectorizer.fit_transform([clean_jd])
        feature_names = vectorizer.get_feature_names_out()
        
        scores = tfidf_matrix.toarray()[0]
        top_jd_terms = [feature_names[i] for i in scores.argsort()[::-1][:15]]
        missing_keywords = [kw for kw in top_jd_terms if kw not in clean_resume]

        # Generate targeted resume formatting bullet metrics
        for kw in missing_keywords:
            ai_suggestions.append(
                f"Optimized system infrastructure metrics by natively integrating **{kw.title()}** solutions, improving execution speed by 18%."
            )

    return {
        "match_percentage": score,
        "eligible": score >= 50.0,
        "missing_keywords": missing_keywords,
        "ai_suggestions": ai_suggestions,
        "application_link": job_record["link"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)