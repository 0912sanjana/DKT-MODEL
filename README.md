# 🧠 Deep Knowledge Tracing (DKT) Model

This repository contains a working implementation of a **Deep Knowledge Tracing (DKT)** model using LSTM to predict student mastery in an adaptive learning environment.  
The goal of this model is to estimate how well a student understands topics and predict whether they will answer the next question correctly based on past performance.

---

## 🚀 Features

- ✅ LSTM-based Deep Knowledge Tracing
- ✅ Predict student performance in real-time
- ✅ Sequence-based student learning analysis
- ✅ Easy to train & extend
- ✅ Ready for LMS integration (Flask / FastAPI / React)
- ✅ Supports student analytics dashboards

---

## 📂 Project Structure


---

## ⚙️ Installation

### Clone this repository
```bash
git clone https://github.com/0912sanjana/DKT-MODEL.git
cd DKT-MODEL

Install dependencies
pip install -r requirements.txt

🧾 Dataset Format

The input dataset must follow the structure:

student_id	question_id	correct	timestamp
S1	Q1	1	2024-01-01 10:30
S1	Q2	0	2024-01-01 10:32
S2	Q1	1	2024-01-02 09:00

correct → 1 = correct, 0 = wrong
Data should be sorted by timestamp per student
Place your dataset inside data/ folder.

▶️ Train the Model
Run:
python scripts/train_dkt.py
After training, the model will be saved automatically in results/.

📊 Output / Results

✅ Loss curve during training
✅ Predicted mastery scores per student
✅ Accuracy improvement over time
🎯 Saved model weights: results/saved_model.pth

This model can now be used inside LMS / adaptive exam systems / EdTech platforms.

🧠 What is Deep Knowledge Tracing?

Deep Knowledge Tracing (DKT) is a sequence model that:

Learns how student knowledge evolves over time
Uses past question results to predict future answers
Helps track mastery level per student/topic
Used in EdTech tools like:
Intelligent Tutoring Systems (ITS)
Adaptive exams / practice apps
Personalized education systems


🏃‍♂️ How to Run the DKT Backend (Full Project)

✅ 1. Clone the Repository
git clone https://github.com/0912sanjana/DKT-MODEL.git
cd DKT-MODEL

✅ 2. Create a Virtual Environment
python -m venv venv

✅ 3. Install Dependencies
pip install -r requirements.txt

✅ 4. Prepare Dataset
Place your dataset file inside data/ folder (example: data/dataset.csv)

✅ 5. Train the Model (Optional)
python scripts/train_dkt.py

This will generate the trained model file:
results/saved_model.pth

✅ 6. Run Backend Server

If your backend file is:
backend/app.py

Run:
python backend/app.py


