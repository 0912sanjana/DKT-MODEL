'''
from models import db_session, StudentResult, Student, Question

def view_results():
    results = db_session.query(StudentResult).all()
    for r in results:
        print(f"Student: {r.student.name}")
        print(f"Question: {r.question.question}")
        print(f"Answer: {r.question.answer}")
        print(f"Correct: {r.is_correct}")
        print(f"Bloom Level: {r.bloom_level}")
        print(f"Topic: {r.question.topic}")
        print(f"DKT Score: {r.dkt_score}")
        print("-" * 40)

if __name__ == "__main__":
    view_results()


$.get(`/get_dashboard?student_id=${student_id}`, function (data) {
    const topicLabels = data.topic_accuracy.map(item => item.topic);
    const topicData = data.topic_accuracy.map(item => item.accuracy);

    const bloomLabels = data.bloom_accuracy.map(item => item.bloom);
    const bloomData = data.bloom_accuracy.map(item => item.accuracy);

    // Topic-wise Chart
    new Chart(document.getElementById("topicChart"), {
      type: "bar",
      data: {
        labels: topicLabels,
        datasets: [{
          label: "Topic Accuracy",
          data: topicData,
          backgroundColor: "skyblue"
        }]
      }
    });

    // Bloom-level Chart
    new Chart(document.getElementById("bloomChart"), {
      type: "bar",
      data: {
        labels: bloomLabels,
        datasets: [{
          label: "Bloom Level Accuracy",
          data: bloomData,
          backgroundColor: "pink"
        }]
      }
       });

    // DKT Score
    document.getElementById("dktScore").innerText =
      `AI-Predicted Mastery Score (DKT): ${data.dkt_score}`;
     });
  }
 @app.route('/submit_answer', methods=['POST'])
def submit_answer():
    session = Session()  # start a safe local session
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        question_id = data.get("question_id")
        user_answer = data.get("user_answer", "").strip().lower()

        q = session.query(Question).filter_by(id=question_id).first()
        if not q:
            return jsonify({"error": "Question not found"}), 404

        is_correct = q.answer.strip().lower() == user_answer
        print("EXPECTED:", q.answer.strip().lower())
        print("USER:", user_answer)

        bloom_level = classify_bloom_level(q.question).strip().capitalize()
        if bloom_level not in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
           bloom_level = "Unknown"

        mastery = estimate_mastery(bloom_level)

        past_results = session.query(StudentResult)\
            .filter_by(student_id=student_id)\
            .order_by(StudentResult.id).all()

        num_questions = get_num_questions(session)  # Make sure this uses session argument
        input_tensor = build_input_sequence(past_results, num_questions)

        dkt_model = DKTModel(input_size=num_questions * 2, hidden_size=50, num_questions=num_questions)
        dkt_model.eval()
        with torch.no_grad():
            dkt_output = dkt_model.predict(input_tensor)
            dkt_score = round(float(dkt_output[0, -1].mean().item()), 2)

        result = StudentResult(
            student_id=student_id,
            question_id=question_id,
            is_correct=is_correct,
            bloom_level=bloom_level,
            dkt_score=str(dkt_score)
        )

        session.add(result)
        session.commit()

        return jsonify({
            "correct": is_correct,
            "bloom_level": bloom_level,
            "mastery_score": mastery,
            "dkt_score": dkt_score
        })

    except Exception as e:
        session.rollback()
        print(f"❌ /submit_answer ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        session.close()  */
        '''

'''from db import SessionLocal
from models import Question, StudentResult
from bloom_classifier import classify_bloom_level

db = SessionLocal()
questions = db.query(Question).filter((Question.bloom_level == None) | (Question.bloom_level == "undefined")).all()

for q in questions:
    q.bloom_level = classify_bloom_level(q.question)

for row in db.query(StudentResult).all():
    print(row.bloom_level)

db.commit()
print("Bloom levels updated.") '''
'''
import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()
cursor.execute("ALTER TABLE questions ADD COLUMN bloom_level TEXT")
conn.commit()
conn.close()

print("✅ bloom_level column added.")

'''
'''
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///database.db")
conn = engine.connect()

try:
    conn.execute(text("ALTER TABLE student_results ADD COLUMN timestamp TEXT;"))
    print("✅ 'timestamp' column added (no default).")
except Exception as e:
    print("⚠️", e)

# Step 2: Update existing rows with current timestamp
try:
    conn.execute(text("UPDATE student_results SET timestamp = datetime('now');"))
    print("✅ All existing timestamps updated.")
except Exception as e:
    print("⚠️", e)
'''
from models import Session, Question

# Create a session
session = Session()

try:
    # Fetch first 5 questions
    questions = session.query(Question).limit(5).all()

    # Print the answers
    for q in questions:
        print(f"Question ID: {q.id}")
        print(f"Question   : {q.question}")
        print(f"Answer     : {q.answer}")
        print("-" * 40)

finally:
    session.close()
