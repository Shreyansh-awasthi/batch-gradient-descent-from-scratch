# batch-gradient-descent-from-scratch
# 📉 Gradient Descent from Scratch

> Understanding what happens **inside** Linear Regression — from sklearn's black box to building it with pure NumPy.

---

## 🧠 Why This Project?

Most ML tutorials just call `model.fit()` and move on.

This project takes a different approach — it implements the **same goal** (predicting with linear regression) in three progressively deeper ways, so you can truly understand what gradient descent is doing under the hood.

---

## 🔢 Three Levels of Implementation

### Level 1 — Sklearn LinearRegression
The standard way. Uses the **Normal Equation** to find the exact solution in one step. No iterations, no learning rate. Great baseline.

### Level 2 — Sklearn SGDRegressor
Uses **Stochastic Gradient Descent** — updates weights one sample at a time. Includes built-in regularization and adaptive learning rate. Closer to how deep learning optimizers work.

### Level 3 — Batch Gradient Descent (Pure NumPy)
Built entirely from scratch. Updates weights using the **full dataset** every epoch.

```python
# The core of gradient descent — just 2 lines
gradient = (1/m) * X.T.dot(X.dot(theta) - y)
theta = theta - learning_rate * gradient
```

This is where the real learning happens.

---

## 📊 Why Are the R² Scores Different?

A common question — and a great one. The scores differ because:

| Factor | LinearRegression | SGDRegressor | Batch GD (Scratch) |
|--------|-----------------|--------------|-------------------|
| Method | Normal Equation | Stochastic updates | Full batch updates |
| Regularization | None | L2 (default) | None |
| Learning Rate | N/A | Adaptive | Fixed |
| Feature Scaling | Not required | Required | Required |

This is **expected behavior** — not a bug. Each optimizer takes a different path to minimize the loss.

---

## 💡 Key Concepts You'll Learn

- How gradient descent minimizes the cost function step by step
- Why feature scaling (normalization) matters for convergence
- The tradeoff between Batch GD, SGD, and Mini-batch GD
- How learning rate affects convergence speed and stability
- Why sklearn's implementations score differently from scratch ones

---

## 🚀 How to Run

**On Kaggle** — just fork the notebook and run all cells.

**Locally:**
```bash
git clone https://github.com/yourusername/gradient-descent-from-scratch.git
cd gradient-descent-from-scratch
pip install numpy pandas matplotlib scikit-learn
jupyter notebook
```

---

## 🗂️ What's Inside the Notebook

```
1. Data Loading & Exploration
2. Feature Scaling with StandardScaler
3. Linear Regression — sklearn baseline
4. SGDRegressor — sklearn optimizer
5. Batch Gradient Descent — from scratch
6. Loss curve visualization
7. R² score comparison across all three
```

---

## 📌 Feature Scaling — Also Built from Scratch

Instead of using sklearn's `StandardScaler`, this project implements feature scaling manually using pure NumPy:

```python
# Custom Standard Scaler from scratch
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std
```

This means **every single step** — from scaling to prediction — is built from the ground up. No black boxes.

Without scaling, gradient descent converges slowly or diverges entirely.

---

## 🎯 Who Is This For?

- ML beginners who want to go beyond just using sklearn
- Anyone preparing for **ML interviews** (this is a very common interview question)
- Students who want to understand the math behind the models

---

## ⭐ Star This Repo

If this helped you understand gradient descent better, consider starring the repo — it helps others find it too!
