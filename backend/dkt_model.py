import torch
import torch.nn as nn
import torch.nn.functional as F
from models import db_session, Question

# ========================
# 🎯 Deep Knowledge Tracing (LSTM)
# ========================
class DKTModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(DKTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out).squeeze(-1))

    def predict(self, sequence_tensor):
        """
        Predict student knowledge state.
        sequence_tensor: shape (1, time_steps, input_size)
        Returns: mastery scores per question at each time step.
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(sequence_tensor)
        return predictions


# ========================
# 🧪 Simulate dummy input (for test/demo)
# ========================
def simulate_student_sequence():
    num_questions = 10
    input_size = num_questions * 2
    hidden_size = 50

    model = DKTModel(input_size=input_size, hidden_size=hidden_size, num_questions=num_questions)

    dummy_input = torch.rand(1, 5, input_size)  # (1 student, 5 time steps, input_size dim)
    output = model.predict(dummy_input)

    print("Simulated output shape:", output.shape)  # (1, 5, 10)
    print("Output (last step):", output[0, -1])     # Mastery scores

from sentence_transformers  import SentenceTransformer, util

# Load model once (very fast)
sbert = SentenceTransformer('all-MiniLM-L6-v2')

def compute_dkt_score(student_answer, expected_answer):
    embedding_student = sbert.encode(student_answer, convert_to_tensor=True)
    embedding_expected = sbert.encode(expected_answer, convert_to_tensor=True)
    similarity = util.cos_sim(embedding_student, embedding_expected).item()
    return round(similarity, 2)  # score between 0 and 1

# ========================
# ✅ Real LMS Data Utils
# ========================
def get_num_questions(session):
    """
    Retrieves the max question ID from DB to define num_questions dynamically.
    """
    return max([q.id for q in session.query(Question).all()] or [0])


def build_input_sequence(results, num_questions):
    """
    Builds time-series tensor input for DKT from student answer logs.
    Each vector: 2N size one-hot encoding [correct@qid | incorrect@qid+N]
    """
    input_size = num_questions * 2
    sequence = []

    for r in results:
        vec = [0.0] * input_size
        idx = r.question_id - 1
        if idx < 0 or idx >= num_questions:
            continue
        pos = idx if r.is_correct else idx + num_questions
        vec[pos] = 1.0
        sequence.append(vec)

    if not sequence:
        sequence.append([0.0] * input_size)

    return torch.tensor([sequence], dtype=torch.float32)


# Run test
if __name__ == "__main__":
    simulate_student_sequence()
