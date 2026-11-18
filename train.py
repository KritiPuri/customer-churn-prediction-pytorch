import argparse
import json
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from preprocessing import build_preprocessor, prepare_features
from model import ChurnANN

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch ANN for customer churn prediction.")
    parser.add_argument("--data", default="Churn_Modelling.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    set_seed(42)
    os.makedirs("artifacts", exist_ok=True)

    data = pd.read_csv(args.data)
    X = prepare_features(data)
    y = data["Exited"].astype("float32").to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_test = preprocessor.transform(X_test).astype("float32")
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

    # Validation split is taken only from the training data.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    def loader(x, target, shuffle):
        ds = TensorDataset(torch.tensor(x), torch.tensor(target).reshape(-1, 1))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(X_fit, y_fit, True)
    val_loader = loader(X_val, y_val, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnANN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best_loss, best_state, stale = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += criterion(model(xb), yb).item() * len(xb)
        val_loss = val_total / len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().ravel()
        probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "best_validation_loss": float(best_loss),
        "epochs_trained": len(history["train_loss"]),
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": int(X_train.shape[1]),
        "hidden1": 64, "hidden2": 32, "dropout": 0.2,
        "threshold": 0.5,
    }, "artifacts/churn_model.pth")
    with open("artifacts/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open("artifacts/training_history.json", "w") as f: json.dump(history, f, indent=2)
    print("\nTest metrics:")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
