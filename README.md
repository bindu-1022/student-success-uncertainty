# Student Success Prediction with Uncertainty Quantification

> Predict student outcomes with confidence — not just accuracy.

---

## What This Project Does

Most machine learning models answer one question: *"What will happen?"*  
This project answers three: **What? How confident? Why?**

By combining a Random Forest classifier with bootstrap-based uncertainty estimation and feature importance analysis, this system gives educators and students actionable, trustworthy predictions — not just a number.

---

## The Problem

Standard ML models output a prediction, but give no indication of how much to trust it. In academic settings, a low-confidence prediction of "Pass" is very different from a high-confidence one. Acting on uncertain predictions without knowing they're uncertain leads to poor decisions.

**This project solves that** by quantifying uncertainty alongside every prediction, and explaining which factors drove the result.

---

## How It Works

### 1. Data

Uses the [Student Performance Dataset](https://www.kaggle.com/) with the following input features:

| Feature | Description |
|---|---|
| `studytime` | Weekly study hours |
| `failures` | Number of past class failures |
| `absences` | Number of school absences |
| `G1` | First period grade |
| `G2` | Second period grade |

**Target:** Pass / Fail — derived from the final grade `G3` (pass if G3 ≥ 10).

---

### 2. Model

- **Algorithm:** Random Forest Classifier
- **Split:** 80% training / 20% testing
- **Accuracy:** ~90%

---

### 3. Uncertainty Quantification

Instead of a single model prediction, the system runs **50 bootstrap samples** of the training data and trains a model on each:

- **Mean prediction** across all 50 runs → the final pass probability
- **Standard deviation** across all 50 runs → the uncertainty (±%)

A high standard deviation means the model is unsure. A low one means it's confident. This distinction matters.

---

### 4. Explainability

Feature importance scores are computed from the Random Forest to show *why* a prediction was made — surfacing which student attributes had the most influence on the outcome.

---

## Dashboard

The interactive Streamlit dashboard includes:

- **Input sliders** for all 5 features
- **Pass probability** display
- **Uncertainty band** visualization (±%)
- **Risk level classification** (Low / Medium / High)
- **Feature importance bar chart**

### Example Output

| Metric | Value |
|---|---|
| Pass Probability | 30% |
| Uncertainty | ± 9% |
| Risk Level | 🔴 High Risk |

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard
├── model.py                # Random Forest + bootstrap training
├── uncertainty.py          # Uncertainty quantification logic
├── explainability.py       # Feature importance analysis
├── data/
│   └── student-mat.csv     # Student performance dataset
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/student-success-prediction.git
cd student-success-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Key Concepts

**Uncertainty Quantification (UQ)** — The process of estimating how confident a model is in its prediction. Here, bootstrap sampling creates a distribution of predictions; the spread of that distribution is the uncertainty.

**Explainable AI (XAI)** — Techniques that make model decisions interpretable. Feature importance tells you which inputs the model relied on most, making predictions auditable and actionable.

---

## Why This Matters

A model that says *"70% chance of passing — but I'm only 60% confident in that"* is far more useful than one that silently outputs `0.7`. This project demonstrates that uncertainty and explainability aren't optional add-ons — they're essential components of responsible ML in high-stakes domains like education.

---

## License

MIT License. See `LICENSE` for details.
