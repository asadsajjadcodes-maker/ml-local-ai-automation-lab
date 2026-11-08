<div align="center">

# Machine Learning → Deep Learning → AI Engineering

### A Practical, Code-First Machine Learning & AI Engineering Lab

Building end-to-end mastery in machine learning, deep learning, and production AI engineering by implementing core algorithms from scratch in **Python** and **NumPy**, transitioning to industry-standard libraries like **Scikit-Learn** and **PyTorch**, and serving models with **FastAPI**.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)](https://matplotlib.org/)
[![Learning](https://img.shields.io/badge/Status-Day_25_Completed-2ea44f?style=for-the-badge)](https://github.com/asadsajjadcodes-maker/ml-local-ai-automation-lab)

[Repository](https://github.com/asadsajjadcodes-maker/ml-local-ai-automation-lab) · [My GitHub Profile](https://github.com/asadsajjadcodes-maker)

</div>

---

## 📖 About This Repository

This repository documents my step-by-step engineering progression from foundational machine learning mathematics to deep learning architectures and production AI systems. Every concept is built from first principles before adopting high-level abstractions, ensuring deep mathematical and practical intuition.

- **Phase 1 Target Deadline:** January 20, 2027
- **Long-Term Roadmap:** 2-Year Advanced AI Engineering Mastery Track (through January 2029)

---

## 📈 Progress at a Glance

**Completed: 25 Learning Days (Up to Day 25)** · **Current Stage: Machine Learning Foundations**

| Day | Topic | Core Implementation & Concepts Practised | Implementation | Status |
|---:|---|---|:---:|:---:|
| **01** | Weights and Predictions | Map input features to target predictions using a linear weight and compute total absolute error | [`day01.py`](src/ml_core/foundations/day_01/day01.py) | ✅ |
| **02** | Learning and Weight Updates | Iteratively refine a model weight using prediction error and a controlled learning rate | [`day02_learning_weight.py`](src/ml_core/foundations/day_02/day02_learning_weight.py) | ✅ |
| **03** | Error, Squared Error, and MSE | Quantify model performance by computing residuals, squared errors, and Mean Squared Error (MSE) | [`loss.py`](src/ml_core/foundations/day_03/loss.py) | ✅ |
| **04** | Gradient Descent | Derive analytical gradients of squared loss and update parameters opposite to the gradient slope | [`gradient_descent.py`](src/ml_core/foundations/day_04/gradient_descent.py) | ✅ |
| **05** | Multiple Training Examples | Aggregate and average losses and gradients across a full batch of training examples | [`multiple_examples.py`](src/ml_core/foundations/day_05/multiple_examples.py) | ✅ |
| **06** | Linear Regression Training | Train a full univariate linear regression model ($y = wx + b$) over 3,500 epochs and run inference | [`linear_regression.py`](src/ml_core/foundations/day_06/linear_regression.py) | ✅ |
| **07** | NumPy Foundations | Replace Python loops with vectorized NumPy array broadcasting for forward pass and MSE calculation | [`numpy_basics.py`](src/ml_core/foundations/day_07/numpy_basics.py) | ✅ |
| **08** | Multiple Features and Dot Products | Combine multiple input features into 2D matrices and compute vectorized predictions via `np.dot` | [`multiple_features.py`](src/ml_core/foundations/day_08/multiple_features.py) | ✅ |
| **09** | Training a Multi-Feature Model | Optimize multiple feature weights and bias simultaneously using matrix transpose gradient descent | [`multi_feature_training.py`](src/ml_core/foundations/day_09/multi_feature_training.py) | ✅ |
| **10** | Feature Scaling | Apply column-wise Min-Max normalization to multi-scale features for stable gradient convergence | [`feature_scaling.py`](src/ml_core/foundations/day_10/feature_scaling.py) | ✅ |
| **11** | Train/Test Split | Shuffle datasets deterministically (`seed=42`), partition 80/20 train/test splits, and evaluate generalization | [`train_test_split.py`](src/ml_core/foundations/day_11/train_test_split.py) | ✅ |
| **12** | Overfitting vs. Underfitting | Build diagnostic heuristics comparing training MSE and testing MSE to detect high bias or high variance | [`overfitting.py`](src/ml_core/foundations/day_12/overfitting.py) | ✅ |
| **13** | Validation and Model Selection | Compare candidate models systematically and select the optimal architecture via lowest validation MSE | [`model_selection.py`](src/ml_core/foundations/day_13/model_selection.py) | ✅ |
| **14** | Data Leakage and Scaling Safely | Prevent data leakage by fitting Min-Max scalers strictly on the 70% training split before transforming val/test sets | [`leakage_and_scaling.py`](src/ml_core/foundations/day_14/leakage_and_scaling.py) | ✅ |
| **15** | Complete ML Workflow (Train/Val/Test) | End-to-end pipeline: shuffle, split (70/15/15), zero-safe scaling, multi-model selection, and held-out test evaluation | [`complete_ml_workflow.py`](src/ml_core/foundations/day_15/complete_ml_workflow.py) | ✅ |
| **16** | Model Evaluation: MSE, MAE, and RMSE | Implement MAE, MSE, and RMSE from scratch in NumPy and analyze metric sensitivity to extreme outliers | [`mae_mse_rmse.py`](src/ml_core/foundations/day_16/mae_mse_rmse.py) | ✅ |
| **17** | Debugging Gradient Descent & Target Scaling | Normalize both input features ($X$) and target values ($y$) without leakage, select models, and invert scaling for dollar-denominated predictions | [`debugging_gradient_descent.py`](src/ml_core/foundations/day_17/debugging_gradient_descent.py) | ✅ |
| **18** | Regression Metrics with Scikit-Learn | Benchmark regression predictions and outlier impact using `scikit-learn`'s `mean_absolute_error`, `mean_squared_error`, and `root_mean_squared_error` | [`regression_metrics.py`](src/ml_core/foundations/day_18/regression_metrics.py) | ✅ |
| **19** | Residual and Error Analysis | Compute individual residuals ($y - \hat{y}$), analyze directional bias with mean residual & evaluation metrics (MAE, MSE, RMSE), and plot residual patterns via Matplotlib | [`residual_analysis.py`](src/ml_core/foundations/day_19/residual_analysis.py) | ✅ |
| **20** | Feature Engineering | Construct domain-specific ratios, polynomial transformations, and feature interactions; combine matrices via `np.column_stack` | [`feature_engineering.py`](src/ml_core/foundations/day_20/feature_engineering.py) | ✅ |
| **21** | Categorical Features & One-Hot Encoding | Separate numeric and categorical columns, encode multi-column categories with Scikit-Learn's `OneHotEncoder` (`handle_unknown="ignore"`, `sparse_output=False`), and assemble the unified feature matrix via `np.column_stack` | [`one_hot_encoding.py`](src/ml_core/foundations/day_21/one_hot_encoding.py) | ✅ |
| **22** | Handling Missing Values & Imputation | Detect missing values with `np.isnan`, prevent data leakage by fitting `SimpleImputer` (median strategy) strictly on training data, and transform test sets consistently | [`missing_values.py`](src/ml_core/foundations/day_22/missing_values.py) | ✅ |
| **23** | End-to-End Preprocessing Pipelines | Build modular `Pipeline` workflows with `ColumnTransformer` (numeric imputation & standard scaling, categorical imputation & one-hot encoding), fit strictly on training data, and train `LinearRegression` | [`preprocessing_pipeline.py`](src/ml_core/foundations/day_23/preprocessing_pipeline.py) | ✅ |
| **24** | Linear Regression with Scikit-Learn | Train Scikit-Learn's `LinearRegression` on train/test splits, inspect learned parameters (`coef_`, `intercept_`), evaluate test performance via MAE & MSE, and predict on unseen inputs | [`linear_regression_sklearn.py`](src/ml_core/foundations/day_24/linear_regression_sklearn.py) | ✅ |
| **25** | Multiple Linear Regression with Scikit-Learn | Train a multivariate `LinearRegression` model on multi-feature data (size, bedrooms, age), inspect weight vectors (`coef_`) and bias (`intercept_`), evaluate generalization via MAE & MSE, and predict on unseen multivariate samples | [`multiple_linear_regression.py`](src/ml_core/foundations/day_25/multiple_linear_regression.py) | ✅ |
| **26** | Polynomial Features & Non-Linear Regression | Transform single features into higher-order polynomial spaces via `PolynomialFeatures` to capture non-linear quadratic relationships, train `LinearRegression`, and evaluate predictions | [`polynomial_regression.py`](src/ml_core/foundations/day_26/polynomial_regression.py) | ✅ |
| **27** | Model Complexity & Bias-Variance Trade-Off | Evaluate polynomial models across multiple complexity degrees ($d \in [1, 20]$), benchmark train vs. test MSE, and analyze the transition from underfitting to optimal fit and severe overfitting | [`model_complexity.py`](src/ml_core/foundations/day_27/model_complexity.py) | ✅ |

---

## 🗂️ Repository Structure

```text
ml-local-ai-automation-lab/
├── src/
│   └── ml_core/
│       └── foundations/
│           ├── day_01/   # Linear predictions & absolute loss
│           ├── day_02/   # Iterative weight learning
│           ├── day_03/   # Squared error & Mean Squared Error (MSE)
│           ├── day_04/   # Single-feature Gradient Descent
│           ├── day_05/   # Batch Gradient Descent across multiple examples
│           ├── day_06/   # Full Linear Regression (weight + bias) & inference
│           ├── day_07/   # Vectorized operations with NumPy arrays
│           ├── day_08/   # Multivariate features & matrix dot products
│           ├── day_09/   # Multi-feature Linear Regression training
│           ├── day_10/   # Min-Max feature normalization
│           ├── day_11/   # Reproducible shuffling & Train/Test evaluation
│           ├── day_12/   # Diagnosing Overfitting vs. Underfitting
│           ├── day_13/   # Validation-driven model selection
│           ├── day_14/   # Leakage-free preprocessing pipelines
│           ├── day_15/   # End-to-end Train/Val/Test ML workflow
│           ├── day_16/   # MAE, MSE & RMSE from scratch + outlier analysis
│           ├── day_17/   # Target scaling, gradient debugging & inverse transforms
│           ├── day_18/   # Standardized regression metrics with Scikit-Learn
│           ├── day_19/   # Residual and error analysis with Matplotlib diagnostics
│           ├── day_20/   # Feature engineering: ratios, polynomial & interaction terms
│           ├── day_21/   # Categorical encoding with OneHotEncoder & matrix assembly
│           ├── day_22/   # Missing data handling & imputation with SimpleImputer
│           ├── day_23/   # Full preprocessing pipelines with ColumnTransformer & Pipeline
│           ├── day_24/   # Scikit-Learn Linear Regression: fitting, coef_/intercept_ & inference
│           ├── day_25/   # Multivariate Linear Regression: multi-feature fitting, weights vector & inference
│           ├── day_26/   # Polynomial regression: non-linear feature transformation & evaluation
│           └── day_27/   # Model complexity: polynomial degrees & bias-variance trade-off
├── .gitattributes
├── .gitignore
└── README.md
```

---

## 🧠 8-Step Engineering Discipline

For every concept introduced in this lab, I follow a structured 8-step engineering workflow:

1. **Quest Briefing** — Understand why the concept matters in real-world production AI systems.
2. **Simple Explanation** — Build first-principles intuition before touching formulas.
3. **Manual Calculation** — Work through the mathematics by hand with small, concrete numbers.
4. **Guided Coding** — Design clean, readable implementations with clear variable semantics.
5. **Independent Challenge** — Implement, experiment, debug edge cases, and verify outputs.
6. **Boss Battle** — Stress-test conceptual and practical mastery on unseen problems.
7. **GitHub Quest** — Run verification checks, document insights, and commit cleanly.
8. **Progression** — Review and reinforce foundations before advancing to the next milestone.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **NumPy**, **Pandas**, **Scikit-Learn**, & **Matplotlib**

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Running Any Daily Experiment

Navigate to the repository root and execute any script directly:

```bash
python src/ml_core/foundations/day_26/polynomial_regression.py
python src/ml_core/foundations/day_27/model_complexity.py
```

---

<div align="center">

**Progress: 27 Learning Days Completed**  
*Learn the concept. Implement it. Test it. Understand why it works.*

</div>
