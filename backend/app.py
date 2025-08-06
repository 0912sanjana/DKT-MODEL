from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from models import init_db, Student, Question, StudentResult, db_session, Session
from bloom_classifier import classify_bloom_level
from bayesian_engine import estimate_mastery
from dkt_model import DKTModel, build_input_sequence, get_num_questions
import torch
import random
from sqlalchemy import Integer, Float
from datetime import datetime
import matplotlib.pyplot as plt
import traceback
import io
import base64
from sentence_transformers import SentenceTransformer, util
import csv
from difflib import SequenceMatcher

app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

# ✅ Initialize DB
init_db()

model = SentenceTransformer('all-MiniLM-L6-v2')
# ----------------- ROUTES -----------------

@app.route('/')
def home():
    return render_template("login.html")

@app.route("/quiz")
def quiz_page():
    return render_template("quiz.html")

@app.route('/dashboard')
def dashboard_page():
    return render_template("dashboard.html")

@app.route('/leaderboard')
def leaderboard_page():
    return render_template("leaderboard.html")

# ----------------- API ENDPOINTS -----------------

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400

    student = Student(name=name)
    db_session.add(student)
    db_session.commit()

    return jsonify({"student_id": student.id, "name": student.name})

@app.route('/get_quiz', methods=['GET'])
def get_quiz():
    topic = request.args.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    questions = Session.query(Question).filter_by(topic=topic).all()
    selected = random.sample(questions, min(len(questions), 5))
    return jsonify([
        {"id": q.id, "question": q.question, "topic": q.topic, "expected_answer": q.answer, "bloom_level": q.bloom_level}
        for q in selected
    ])

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    session = Session()
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        question_id = data.get("question_id")
        user_answer = data.get("user_answer", "").strip().lower()

        q = session.query(Question).filter_by(id=question_id).first()
        if not q:
            return jsonify({"error": "Question not found"}), 404

        correct_answer = q.answer.strip().lower()

        # ✅ Use SentenceTransformer to compute similarity
        embeddings = model.encode([correct_answer, user_answer])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        is_correct = similarity >= 0.40  # adjust this threshold as needed

        # ✅ Bloom classification
        bloom_level = classify_bloom_level(q.question).strip().capitalize()
        if bloom_level not in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
            bloom_level = "Unknown"

        # ✅ Estimate DKT Score
        mastery = estimate_mastery(bloom_level)
        past_results = session.query(StudentResult).filter_by(student_id=student_id).order_by(StudentResult.id).all()
        num_questions = get_num_questions(session)
        input_tensor = build_input_sequence(past_results, num_questions)

        try:
            dkt_model = DKTModel(input_size=num_questions * 2, hidden_size=64)
            dkt_model.eval()
            with torch.no_grad():
                dkt_output = dkt_model.predict(input_tensor)
                dkt_score = round(float(dkt_output[0, -1].mean().item()), 2)
        except Exception as dkt_error:
            dkt_score = 50.0
            print("⚠️ DKT model load error:", dkt_error)

        # ✅ Save to DB
        result = StudentResult(
            student_id=student_id,
            question_id=question_id,
            is_correct=is_correct,
            dkt_score=dkt_score,
            bloom_level=bloom_level,
            timestamp=datetime.utcnow(),
        )

        # ✅ Debug Logs
        print("\n✅ DEBUG: Answer Submission")
        print(f"Student ID: {student_id}")
        print(f"Question ID: {question_id}")
        print(f"Student Answer: {user_answer}")
        print(f"Expected Answer: {correct_answer}")
        print(f"🔍 Similarity Score: {similarity}")
        print(f"Is Correct? {is_correct}")
        print(f"Bloom Level: {bloom_level}")
        print(f"Topic: {q.topic}")
        print(f"DKT Score: {dkt_score}")

        session.add(result)
        session.commit()

        return jsonify({
            "correct": is_correct,
            "bloom_level": bloom_level,
            "mastery_score": mastery,
            "dkt_score": dkt_score,
            "expected_answer": correct_answer
        })

    except Exception as e:
        session.rollback()
        print("❌ Exception occurred in /submit_answer:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()

@app.route('/api/leaderboard')
def get_leaderboard():
    from sqlalchemy.sql import func

    results = db_session.query(
        Student.id,
        Student.name,
        func.count(StudentResult.id).label("total"),
        func.sum(StudentResult.is_correct.cast(Integer)).label("correct"),
        func.avg(StudentResult.dkt_score.cast(Float)).label("avg_dkt")
    ).join(StudentResult).group_by(Student.id).all()

    leaderboard = []
    for r in results:
        accuracy = round((r.correct / r.total) * 100, 2) if r.total else 0
        leaderboard.append({
            "name": r.name,
            "accuracy": accuracy,
            "avg_dkt": round(r.avg_dkt * 100, 2) if r.avg_dkt else 0
        })

    leaderboard = sorted(leaderboard, key=lambda x: (-x["accuracy"], -x["avg_dkt"]))
    return jsonify(leaderboard)

@app.route('/get_dashboard', methods=['GET'])
def get_dashboard():
    student_id = request.args.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id required"}), 400

    student_id = int(student_id)
    results = db_session.query(StudentResult).filter_by(student_id=student_id).all()

    print("\n📊 DEBUG: Dashboard Generation for Student ID:", student_id)
    print(f"Total results fetched: {len(results)}")

    # Topic-wise Accuracy
    topic_scores = {}
    for r in results:
        topic = r.question.topic if r.question else "unknown"
        topic_scores.setdefault(topic, {"correct": 0, "total": 0})
        topic_scores[topic]["total"] += 1
        if r.is_correct:
            topic_scores[topic]["correct"] += 1

    topic_accuracy = []
    print("\n📌 Topic-wise Accuracy:")
    for topic, scores in topic_scores.items():
        accuracy = round(scores["correct"] / scores["total"] * 100, 2)
        topic_accuracy.append({"topic": topic, "accuracy": accuracy})
        print(f"• Topic: {topic} | Accuracy: {accuracy}% ({scores['correct']}/{scores['total']})")

    # Bloom-level Accuracy
    bloom_scores = {}
    for r in results:
        bloom = r.bloom_level if r.bloom_level else "undefined"
        bloom_scores.setdefault(bloom, {"correct": 0, "total": 0})
        bloom_scores[bloom]["total"] += 1
        if r.is_correct:
            bloom_scores[bloom]["correct"] += 1

    bloom_accuracy = []
    print("\n🧠 Bloom-Level Accuracy:")
    for bloom, scores in bloom_scores.items():
        accuracy = round(scores["correct"] / scores["total"] * 100, 2)
        bloom_accuracy.append({"bloom": bloom, "accuracy": accuracy})
        print(f"• Bloom: {bloom} | Accuracy: {accuracy}% ({scores['correct']}/{scores['total']})")

    # DKT Scores
    dkt_scores = [float(r.dkt_score) for r in results if r.dkt_score is not None]
    avg_dkt = round(sum(dkt_scores) / len(dkt_scores), 2) if dkt_scores else None
    dkt_timeline = dkt_scores
    print("\n📈 DKT Scores:")
    print("All DKT scores:", dkt_scores)
    print("Average DKT Score:", avg_dkt)

    # Plot DKT Over Time
    results = db_session.query(StudentResult).filter_by(student_id=student_id).order_by(StudentResult.timestamp).all()
    timestamps = [res.timestamp for res in results if res.dkt_score is not None]
    scores = [float(res.dkt_score) for res in results if res.dkt_score is not None]

    plt.figure(figsize=(6, 3))
    plt.plot(timestamps, scores, marker='o', color='blue')
    plt.xticks(rotation=45)
    plt.title("DKT Mastery Over Time")
    plt.xlabel("Time")
    plt.ylabel("Mastery (%)")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "topic_accuracy": topic_accuracy,
        "bloom_accuracy": bloom_accuracy,
        "dkt_score": avg_dkt,
        "dkt_timeline": dkt_timeline,
        "timestamps": [ts.strftime("%b %d, %H:%M") for ts in timestamps],  # formatted
        "plot_url": None
    })

@app.route('/export_data', methods=['GET'])
def export_data():
    results = db_session.query(StudentResult).all()

    def generate():
        header = ["Student", "Question", "Answer", "Correct", "Bloom Level", "Topic", "DKT Score"]
        yield ",".join(header) + "\n"

        for r in results:
            row = [
                r.student.name,
                r.question.question,
                r.question.answer,
                "Yes" if r.is_correct else "No",
                r.bloom_level,
                r.question.topic,
                str(r.dkt_score) if r.dkt_score is not None else "N/A"
            ]
            yield ",".join(row) + "\n"

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment; filename=student_results.csv"})

@app.teardown_appcontext
def shutdown_session(exception=None):
    Session.remove()

# ----------------- MAIN -----------------

if __name__ == '__main__':
    app.run(port=8000, debug=True)
