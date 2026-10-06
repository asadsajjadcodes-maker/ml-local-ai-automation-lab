# Machine Learning → Deep Learning → AI Engineering

A practical, code-first learning journey from Machine Learning fundamentals to Deep Learning and AI Engineering.

**Target completion date:** January 20, 2027

The goal is not just to watch theory. Every lesson includes:
- Simple explanations
- Python/NumPy practice
- Manual calculations
- Experiments
- Real-world connections
- GitHub commits
- Small challenges to make the concepts stick

---

## 🚀 Progress

| Day | Topic | Status |
|---|---|---|
| Day 1 | Understanding Weights and Predictions | ✅ Complete |
| Day 2 | Learning and Weight Updates | ✅ Complete |
| Day 3 | Error, Squared Error and MSE | ✅ Complete |
| Day 4 | Gradient Descent | ✅ Complete |
| Day 5 | Multiple Examples / Batch Gradient Descent | ✅ Complete |
| Day 6 | Learning Rate and Training Steps | ✅ Complete |
| Day 7 | NumPy for Machine Learning | ✅ Complete |
| Day 8 | Multiple Features + Dot Product | ✅ Complete |
| Day 9 | Training a Multi-Feature Model | ✅ Complete |
| Day 10 | Feature Scaling | ✅ Complete |
| Day 11 | Train/Test Split | ⏳ Next |

---

# 🧠 Day 1 — Weights and Predictions

Started with the basic idea behind a machine learning model.

The model uses a **weight** to turn an input into a prediction.

Basic idea:

```text
input × weight = prediction
```

The important lesson was that we don't want to manually choose the perfect weight.

We want the computer to **learn the weight automatically**.

### Key understanding

> The weight helps the model make its prediction closer to the actual answer.

---

# 🧠 Day 2 — Learning

The model started learning instead of using one fixed weight.

The basic cycle became:

```text
Prediction
    ↓
Compare with actual answer
    ↓
Find error
    ↓
Change weight
    ↓
Predict again
```

This introduced the idea that a machine learning model can improve through repeated training.

---

# 🧠 Day 3 — Error, Squared Error and MSE

Learned why simply using error can cause problems because errors can be positive or negative.

Example:

```text
Actual = 20
Prediction = 10

Error = 10
```

But if:

```text
Actual = 10
Prediction = 20

Error = -10
```

Squaring removes the negative sign:

```text
(-10)² = 100
10² = 100
```

Then learned **Mean Squared Error (MSE)**:

```python
mse = np.mean(error ** 2)
```

### Key idea

> MSE measures how wrong the model's predictions are on average.

---

# 🧠 Day 4 — Gradient Descent

Learned how the computer can automatically adjust the weight.

The model asks:

> "Which direction should I move the weight to reduce the error?"

This introduced the **gradient**.

The basic idea:

```text
Calculate prediction
        ↓
Calculate error/loss
        ↓
Calculate gradient
        ↓
Update weight
        ↓
Repeat
```

The learning rate controls how big each update is.

---

# 🧠 Day 5 — Multiple Examples

Moved from training on one example to training on multiple examples.

Learned that instead of looking at one example at a time, we can calculate the total/average loss and gradient across a batch.

Important ideas:

- Multiple training examples
- Total loss
- Average loss
- Total gradient
- Average gradient
- Updating the weight using the average gradient

This was the first step toward thinking about datasets rather than individual examples.

---

# 🧠 Day 6 — Learning Rate and Training Steps

Experimented with different learning rates.

One important discovery was that:

```text
learning_rate = 0.01
```

can require far more than 100 training steps to converge.

In one experiment it took roughly **1400 steps**.

This showed an important real-world lesson:

> There is no magic number of training steps that works for every learning rate and dataset.

A smaller learning rate usually means smaller updates and can require more steps.

---

# 🧠 Day 7 — NumPy

Started using NumPy to perform machine learning calculations efficiently.

Learned:

```python
np.array()
np.mean()
```

and vectorized calculations.

Instead of manually calculating every example, NumPy can process arrays together.

For example:

```python
predictions = hours * weight + bias
error = actual - predictions
loss = error ** 2
mse = np.mean(loss)
```

This was an important transition from manually calculated ML to actual ML code.

---

# 🧠 Day 8 — Multiple Features + Dot Product

Moved from a model with one feature to a model with multiple features.

Instead of:

```text
one feature → one weight
```

we now have:

```text
feature 1 → weight 1
feature 2 → weight 2
feature 3 → weight 3
```

The model can be represented as:

```text
prediction =
feature1 × weight1
+
feature2 × weight2
+
feature3 × weight3
+
bias
```

Learned to use:

```python
np.dot(features, weights)
```

Example:

```python
features = np.array([
    [5, 7, 20],
    [7, 8, 30],
    [8, 6, 40]
])

weights = np.array([5, 2, 1])

bias = 10

predictions = np.dot(features, weights) + bias
```

This introduced the idea behind:

```text
XW + b
```

which is a fundamental operation in machine learning.

---

# 🧠 Day 9 — Training a Multi-Feature Model

Built a model that could **learn multiple weights automatically**.

Dataset:

```text
Feature 1 = study hours
Feature 2 = marks in class test
Target = final exam score
```

The model started with:

```python
weights = np.array([1.0, 1.0])
bias = 0.0
```

Then used gradient descent to learn better values.

Core prediction:

```python
predictions = np.dot(X, weights) + bias
```

Gradient calculation:

```python
weight_gradient = (-2 / n) * np.dot(X.T, error)

bias_gradient = (-2 / n) * np.sum(error)
```

Weight update:

```python
weights = weights - learning_rate * weight_gradient

bias = bias - learning_rate * bias_gradient
```

After training, the model learned approximately:

```text
Weights = [1.33333333, 1.66666667]
Bias    = 0.33333333
```

The predictions became:

```text
Actual:
[5, 8, 11, 14, 17]

Predictions:
[5, 8, 11, 14, 17]
```

This was a major milestone because the model was no longer using manually chosen weights.

It **learned the weights from data**.

---

# 🧠 Day 10 — Feature Scaling

Learned why features with very different numerical ranges can make training harder.

Real-world example:

```text
house size = 2500
bedrooms   = 5
age        = 10
```

The features have very different scales.

Learned **Min-Max Scaling**:

```python
x_scaled = (x - minimum) / (maximum - minimum)
```

This transforms values so they generally fall between:

```text
0 → 1
```

Example:

```python
minimum = X.min(axis=0)
maximum = X.max(axis=0)

X_scaled = (X - minimum) / (maximum - minimum)
```

Also learned:

```python
axis=0
```

means we calculate the minimum/maximum separately down each column.

### Important understanding

Feature scaling does **not** change the real-world meaning of the data.

It changes the numerical representation so the learning algorithm can work with the features more effectively.

---

# 🔥 Current ML Pipeline

After 10 days, the learning pipeline looks like this:

```text
Raw Data
   ↓
Features
   ↓
Feature Scaling
   ↓
Weights + Bias
   ↓
Prediction
   ↓
Error
   ↓
Loss / MSE
   ↓
Gradient
   ↓
Update Weights
   ↓
Repeat
```

This is the foundation we will keep building on.

---

# 🛠️ Tools Used So Far

- Python
- NumPy
- VS Code
- Git
- GitHub
- Fedora Linux

---

# 🎯 Long-Term Roadmap

The complete goal is:

```text
Machine Learning
       ↓
Deep Learning
       ↓
AI Engineering
       ↓
Real AI Projects
```

Target:

**January 20, 2027**

The learning approach will remain practical and code-first rather than theory-only.

---

# ⚔️ Learning Philosophy

The goal is not to memorize formulas.

For every important concept:

```text
Understand the idea
       ↓
Calculate it manually
       ↓
Write Python
       ↓
Experiment
       ↓
Break it
       ↓
Fix it
       ↓
Build something with it
```

The hardest concepts will be broken into small pieces and connected to real-world examples.

---

# 📈 Progress

**Days completed: 10**

**Current stage: Machine Learning Foundations**

**Next:** Day 11 — Train/Test Split and Generalization

---

## 🏆 Mission

> Don't just make the code run.

> Understand **why** the code works.