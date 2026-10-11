<div align="center">

# Machine Learning → Deep Learning → AI Engineering

### A practical, code-first machine learning learning lab

I am building my machine learning foundations by implementing concepts in Python and NumPy, testing ideas with small experiments, and documenting each milestone as I go.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Learning](https://img.shields.io/badge/Status-Active_Learning-2ea44f?style=for-the-badge)](https://github.com/asadsajjadcodes-maker/ml-local-ai-automation-lab)

[Repository](https://github.com/asadsajjadcodes-maker/ml-local-ai-automation-lab) · [My GitHub Profile](https://github.com/asadsajjadcodes-maker)

</div>

---

## About this repository

This repository documents my step-by-step journey from the fundamentals of machine learning toward deep learning and AI engineering. Instead of relying only on theory or high-level libraries, I implement the core ideas with Python and NumPy so I can understand what happens during training.

The examples are intentionally small and educational. They focus on understanding the mathematics, observing how model parameters change, and learning how to evaluate a model on data it has not seen during training.

## Progress at a glance

**Completed: Days 1–13**  ·  **Current stage: Machine Learning Foundations**

| Day | Topic | What I practised | Status |
|---:|---|---|:---:|
| 1 | Weights and predictions | Use a weight to turn an input into a prediction | ✅ |
| 2 | Learning and weight updates | Compare predictions with actual values and adjust a weight | ✅ |
| 3 | Error, squared error, and MSE | Measure prediction error and calculate mean squared error | ✅ |
| 4 | Gradient descent | Calculate gradients and update model parameters | ✅ |
| 5 | Multiple training examples | Calculate errors and gradients across a batch of examples | ✅ |
| 6 | Linear regression training | Connect predictions, loss, learning rate, and repeated updates | ✅ |
| 7 | NumPy foundations | Use arrays and vectorized operations for ML calculations | ✅ |
| 8 | Multiple features and dot products | Calculate predictions from several input features and weights | ✅ |
| 9 | Training a multi-feature model | Learn multiple weights and a bias with gradient descent | ✅ |
| 10 | Feature scaling | Apply min-max scaling to features with different ranges | ✅ |
| 11 | Train/test split | Train on one subset and evaluate on held-out examples | ✅ |
| 12 | Overfitting vs. underfitting | Compare training and test loss to diagnose model behaviour | ✅ |
| 13 | Validation and model selection | Compare validation MSE and select the model with the lowest validation loss | ✅ |

## What I have learned so far

### 1. How a model makes predictions

For a simple linear model, a prediction is calculated from input features, learned weights, and a bias:

\[
\hat{y} = Xw + b
\]

With one feature, this reduces to the familiar relationship between an input, its weight, and the bias. With multiple features, each feature contributes to the prediction through its corresponding weight.

### 2. How training reduces error

I implemented the basic training loop:

1. Calculate predictions.
2. Compare predictions with actual target values.
3. Calculate the loss, using mean squared error (MSE).
4. Calculate gradients for the weights and bias.
5. Update the parameters using a learning rate.
6. Repeat and observe how the loss changes.

The purpose is to understand how training works before depending on a machine learning framework to handle these steps.

### 3. How to evaluate generalisation

The later lessons introduce evaluation beyond the training dataset:

- **Train/test split:** reserve examples to evaluate the trained model.
- **Underfitting:** a model performs poorly even on its training data.
- **Overfitting:** a model performs well on training data but poorly on unseen data.
- **Validation and model selection:** compare candidate models using validation loss rather than choosing the model only by its training performance.

These lessons form the beginning of a more reliable machine learning workflow.

## Repository structure

```text
ml-local-ai-automation-lab/
└── src/
    └── ml core/
        └── foundations/
            ├── day01.py
            ├── day02_learning_weight.py
            ├── day03/
            │   └── loss.py
            ├── day04/
            │   └── gradient_descent.py
            ├── day05/
            │   └── multiple_examples.py
            ├── day06/
            │   └── linear_regression.py
            ├── day 07/
            │   └── mumpy_basics.py
            ├── day 08/
            │   └── multiple_features.py
            ├── day 09/
            │   └── multi_feature_training.py
            ├── day 10/
            │   └── features_scaling.py
            ├── day 11/
            │   └── train_test_split.py
            ├── day 12/
            │   └── overfitting.py
            └── day 13/
                └── model_selection.py
```

Each lesson is kept as a small Python script so I can run it, read the calculations, change parameters, and observe the result.

## Getting started

### Requirements

- Python **3.12 or newer** is recommended for the current examples.
- NumPy
- A terminal and a code editor

### 1. Clone the repository

```bash
git clone https://github.com/asadsajjadcodes-maker/ml-local-ai-automation-lab.git
cd ml-local-ai-automation-lab
```

### 2. Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install NumPy

```bash
python -m pip install --upgrade pip
python -m pip install numpy
```

### 4. Run a lesson

Run commands from the repository root. The quotation marks are important because some existing directory names contain spaces.

```bash
python "src/ml core/foundations/day 11/train_test_split.py"
python "src/ml core/foundations/day 12/overfitting.py"
python "src/ml core/foundations/day 13/model_selection.py"
```

You can run an earlier lesson by passing its script path in the same way.

## Tools and concepts

| Area | Tools / concepts |
|---|---|
| Programming | Python, functions, loops, basic data structures |
| Numerical computing | NumPy arrays, vectorized operations, dot products |
| Model fundamentals | Features, weights, bias, predictions |
| Training | Mean squared error, gradients, gradient descent, learning rate |
| Data preparation | Shuffling, train/test split, min-max feature scaling |
| Evaluation | Training loss, test loss, validation loss, overfitting, underfitting, model selection |
| Development workflow | VS Code, Git, GitHub, small experiments and incremental commits |

## Learning workflow

For each new concept, my goal is to:

**Understand the idea → work through an example → write the Python code → experiment with the parameters → inspect the result → explain the lesson in my own words.**

The priority is genuine understanding, not just getting a script to run.

## Next steps

The next stage is to strengthen evaluation and generalisation skills, then move toward more structured machine learning workflows. Planned topics include:

- More evaluation metrics and validation techniques.
- Classification fundamentals and appropriate evaluation methods.
- Comparing from-scratch implementations with established ML tools such as scikit-learn.
- Small end-to-end projects that combine data preparation, training, and evaluation.
- Deep learning foundations after the core machine learning concepts are solid.

These are learning goals, not claims that the topics have already been completed.

## Long-term direction

```text
Machine Learning Foundations
            ↓
Model Evaluation and Practical ML
            ↓
Deep Learning
            ↓
AI Engineering and Applied Projects
```

My long-term goal is to build a strong practical foundation for developing useful AI-powered tools and applications.

---

<div align="center">

**Progress: 13 learning days completed**  
*Learn the concept. Implement it. Test it. Understand why it works.*

</div>