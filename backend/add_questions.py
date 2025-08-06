import os
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sqlalchemy.orm import sessionmaker
from models import Question, Course
from db import engine
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-prepend")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-prepend").to(device)

# SQLAlchemy session
Session = sessionmaker(bind=engine)
db = Session()

# --- UTILITIES ---

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_questions(text):
    sentences = text.split(".")
    inputs = [
        "generate question: " + s.strip()
        for s in sentences if len(s.strip().split()) > 8
    ]

    questions = []
    for i in tqdm(inputs, desc="Generating Questions"):
        encoding = tokenizer(i, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model.generate(**encoding)
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        questions.append((question, i.replace("generate question: ", "")))
    return questions

def classify_bloom_level(question):
    question = question.lower()
    if question.startswith("what") or question.startswith("define") or "is called" in question:
        return "Remember"
    elif question.startswith("explain") or question.startswith("describe") or "why" in question:
        return "Understand"
    elif question.startswith("how") or "compare" in question:
        return "Apply"
    elif "analyze" in question or "differentiate" in question:
        return "Analyze"
    elif "evaluate" in question or "judge" in question:
        return "Evaluate"
    elif "create" in question or "design" in question:
        return "Create"
    return "Remember"  # fallback

def add_questions_from_pdf(pdf_path, course_name):
    # Ensure course exists or create it
    course = db.query(Course).filter_by(name=course_name.strip()).first()
    if not course:
        course = Course(name=course_name.strip())
        db.add(course)
        db.commit()
        print(f"✅ Created new course: {course_name}")

    # Extract and generate
    text = extract_text_from_pdf(pdf_path)
    qas = generate_questions(text)

    for q, a in qas:
        bloom_level = classify_bloom_level(q)
        question = Question(
            question=q,
            answer=a,
            topic=course_name.lower().strip(),
            bloom_level=bloom_level,
            course_id=course.id
        )
        db.add(question)

    db.commit()
    print(f"✅ {len(qas)} questions added for course: {course_name}")

# --- MAIN ---

if __name__ == "__main__":
    num_files = int(input("How many PDFs do you want to upload? "))
    for i in range(num_files):
        pdf_filename = input(f"Enter PDF filename {i+1} (e.g., constitution_law_course.pdf): ")
        course_title = input(f"Enter course name for {pdf_filename}: ")

        pdf_file_path = os.path.join("backend", "uploads", pdf_filename)
        if not os.path.exists(pdf_file_path):
            print(f"❌ File not found: {pdf_file_path}")
        else:
            add_questions_from_pdf(pdf_file_path, course_title)
