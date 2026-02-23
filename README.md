# ML Algorithm Implementations

A ZeroCourse project for Course 14.1: Supervised Learning (Week 1).

## What You'll Build

Implement three classic machine learning algorithms **from scratch** — no sklearn for the algorithm itself:

| Class | Description |
|-------|-------------|
| `LinearRegression(lr, n_iter)` | Gradient descent linear regression |
| `LinearRegression.fit(X, y)` | Train weights via gradient descent |
| `LinearRegression.predict(X)` | Predict continuous targets |
| `LinearRegression.score(X, y)` | R² coefficient of determination |
| `LogisticRegression(lr, n_iter)` | Binary classification with sigmoid |
| `LogisticRegression._sigmoid(z)` | Sigmoid activation function |
| `LogisticRegression.fit(X, y)` | Train on binary labels |
| `LogisticRegression.predict_proba(X)` | Output probabilities |
| `LogisticRegression.predict(X)` | Output class labels (0/1) |
| `LogisticRegression.score(X, y)` | Classification accuracy |
| `DecisionTree(max_depth, min_samples)` | CART classifier with Gini impurity |
| `DecisionTree._gini(y)` | Compute Gini impurity |
| `DecisionTree._best_split(X, y)` | Find optimal split |
| `DecisionTree.fit(X, y)` | Build tree recursively |
| `DecisionTree.predict(X)` | Predict class labels |
| `DecisionTree.score(X, y)` | Classification accuracy |
| `DecisionTree.tree_depth()` | Actual depth of built tree |

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests (they will all fail initially):
   ```bash
   python -m pytest tests/ --tb=short
   ```

3. Open `lib/ml_algorithms.py` and implement each class.

4. Run the tests again to check your progress.

## Suggested Implementation Order

1. **`LinearRegression`** — simplest, pure gradient descent
   - `__init__` → `fit` (gradient update loop) → `predict` → `score`
2. **`LogisticRegression`** — builds on LinReg, adds sigmoid
   - `_sigmoid` → `__init__` → `fit` → `predict_proba` → `predict` → `score`
3. **`DecisionTree`** — recursive tree building
   - `_gini` → `_majority_class` → `_best_split` → `_build_tree` → `fit` → `_predict_one` → `predict` → `score`

## Tips

- **Linear Regression gradients:** `dw = (2/n) * X.T @ (y_pred - y)`, `db = (2/n) * sum(y_pred - y)`
- **Sigmoid:** `1 / (1 + exp(-z))` — clip z to `[-500, 500]` to avoid overflow
- **Logistic gradients** are the same shape as linear regression but use sigmoid output
- **Gini impurity:** `1 - sum(p_i²)` where `p_i` = proportion of class `i`
- **CART splitting:** Try every unique value of every feature as a threshold, pick the one with lowest weighted Gini
- **R² score:** `1 - SS_res/SS_tot` where `SS_res = sum((y-pred)²)`, `SS_tot = sum((y-mean)²)`
- Use plain Python lists — no numpy required

## Running Tests

```bash
python -m pytest tests/                              # Run all tests
python -m pytest tests/ -v                           # Verbose output
python -m pytest tests/ -k "TestLinearRegression"    # Run only linear regression tests
python -m pytest tests/ -k "TestLogistic"            # Run only logistic regression tests
python -m pytest tests/ -k "TestDecisionTree"        # Run only decision tree tests
```
