"""Small validation-based architecture search for the churn ANN."""
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from preprocessing import prepare_features
from model import ChurnANN

def main():
    data = pd.read_csv("Churn_Modelling.csv")
    X, y = prepare_features(data), data["Exited"].astype("float32").to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    from preprocessing import build_preprocessor
    prep = build_preprocessor()
    X_train = prep.fit_transform(X_train).astype("float32")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=.15, random_state=42, stratify=y_train
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for h1, h2, lr in itertools.product([32, 64, 128], [16, 32], [1e-3, 5e-4]):
        model = ChurnANN(X_fit.shape[1], h1, h2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit).reshape(-1, 1))
        dl = DataLoader(ds, batch_size=128, shuffle=True)
        for _ in range(30):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val).to(device)).cpu().numpy().ravel()
            val_probs = 1 / (1 + np.exp(-val_logits))
            val_loss = -(y_val*np.log(val_probs+1e-8) + (1-y_val)*np.log(1-val_probs+1e-8)).mean()
        results.append({"hidden1": h1, "hidden2": h2, "learning_rate": lr, "validation_loss": float(val_loss)})
        print(results[-1])
    print("\nBest configuration:", min(results, key=lambda row: row["validation_loss"]))

if __name__ == "__main__":
    main()
