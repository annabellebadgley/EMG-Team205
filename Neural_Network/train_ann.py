

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# =========================
# Dataset
# =========================

class WindowDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array [N, T, C]
        y: numpy array [N]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Model
# =========================

class MLPWindowClassifier(nn.Module):
    def __init__(self, T, C, num_classes):
        super().__init__()
        input_dim = T * C

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten [B, T, C]
        return self.net(x)


# =========================
# Training Function
# =========================

def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += Xb.size(0)

        train_acc = correct / total

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/total:.4f} | Acc: {train_acc:.3f}")


# =========================
# Main
# =========================

if __name__ == "__main__":

    # Example dummy data (REPLACE with your EMG windows)
    N = 1000
    T = 400      # timesteps per window
    C = 10       # EMG channels
    K = 8        # number of movement classes

    X = np.random.randn(N, T, C)
    y = np.random.randint(0, K, size=N)

    dataset = WindowDataset(X, y)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLPWindowClassifier(T=T, C=C, num_classes=K)

    train_model(model, train_loader, val_loader=None, epochs=20)

    torch.save(model.state_dict(), "ann_model.pt")