import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cpu")

# Sample Data
data = {
    'student_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'question_id': [1, 2, 3, 1, 4, 5, 2, 3, 6],
    'is_correct': [1, 0, 1, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Encode questions
label_encoder = LabelEncoder()
df['question_encoded'] = label_encoder.fit_transform(df['question_id'])
num_questions = len(label_encoder.classes_)

# Build interaction sequences
sequences = []
for _, group in df.groupby('student_id'):
    seq = []
    for _, row in group.iterrows():
        qid = row['question_encoded']
        correct = row['is_correct']
        interaction = [0] * (num_questions * 2)
        interaction[qid + (num_questions if correct else 0)] = 1
        seq.append(interaction)
    sequences.append(seq)

max_len = max(len(seq) for seq in sequences)
X = np.zeros((len(sequences), max_len, num_questions * 2))
y = np.zeros((len(sequences), max_len))

for i, seq in enumerate(sequences):
    for j, vec in enumerate(seq):
        X[i, j] = vec
        y[i, j] = sum(vec[num_questions:])

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=2, shuffle=True)

# Define Model
class DKTModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(DKTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out).squeeze(-1))

model = DKTModel(input_size=num_questions*2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Train
for epoch in range(10):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

# Save
torch.save(model.state_dict(), "dkt_trained.pt")
print("✅ Saved as dkt_trained.pt")
