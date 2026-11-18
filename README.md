# Customer Churn Prediction using PyTorch

ANN customer-churn classification using **PyTorch**. It predicts whether a bank customer is likely to leave (`Exited = 1`) using the `Churn_Modelling.csv` dataset.

## Project structure

```text
customer_churn_pytorch/
├── app.py
├── model.py
├── preprocessing.py
├── train.py
├── Churn_Modelling.csv
├── requirements.txt
└── artifacts/
    ├── churn_model.pth
    ├── preprocessor.joblib
    ├── metrics.json
    └── training_history.json
```

## 1. Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate it:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train the ANN

Keep `Churn_Modelling.csv` in the project root, then run:

```bash
python train.py
```

Optional training settings:

```bash
python train.py --epochs 150 --batch-size 128 --learning-rate 0.001 --patience 15
```

The script:
1. Removes `RowNumber`, `CustomerId`, and `Surname` because they are identifiers, not useful model inputs.
2. One-hot encodes `Geography` and `Gender`.
3. Standardizes numeric features using `StandardScaler`.
4. Splits the data into 80% training and 20% testing, with stratification.
5. Trains a two-hidden-layer ANN (64 and 32 neurons, ReLU, dropout) with Adam and binary cross-entropy with logits.
6. Uses a validation split and early stopping.
7. Saves the best model, fitted preprocessing pipeline, and test metrics in `artifacts/`.

The preprocessing pipeline is fitted on training data only to avoid data leakage.

## 3. Run the Streamlit app

After training:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Enter the customer's information and click **Predict churn**.

## 4. Evaluation

After training, `artifacts/metrics.json` contains accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and the classification report. Accuracy alone can be misleading for churn data, so review precision, recall, F1, and ROC-AUC as well.

## Model architecture

- Input: preprocessed customer features
- Hidden layer 1: 64 neurons + ReLU + Dropout
- Hidden layer 2: 32 neurons + ReLU + Dropout
- Output: 1 logit
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Decision threshold: 0.50

## Notes

- The supplied original `model.h5` is a Keras model and is **not** used by this PyTorch implementation.
- The `.pth` file is the PyTorch state dictionary checkpoint.
- Re-train with `python train.py` if you change the dataset or preprocessing.

## Optional: hyperparameter experiment

Run `python tune.py` to compare a small set of hidden-layer sizes and learning rates using validation loss. This is a separate experiment and does not overwrite the trained production checkpoint.
