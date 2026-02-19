# Quantitative Economics & ML Implementation Lab 2026
This repository contains from-scratch implementations of core machine learning algorithms, developed as part of a systematic study of statistical learning and optimization.

## 🛠 Progress Tracker

### 1. Splines & Basis Models
*Foundational models using basis expansion and piecewise polynomials.*

| Topic | Key Math / Concept | Status | Implementation |
| :--- | :--- | :---: | :--- |
| **Linear Spline** | Truncated Power Basis: $(x-\xi)_+$ | ✅ | `basis_models/splines.py` |
| **Cubic Spline** | $C^2$ Continuity & Smoothness | ✅ | `basis_models/splines.py` |
| **Smoothing Spline** | Interpolation | ✅ | `basis_models/splines.py` |
| **Natural Spline** | Boundary constraints (Linear at edges) | 🏗️ | |
| **B-splines** | Local support & Numerical stability | ⬜ | |
| **GAMs** | Additive components: $\sum f_j(x_j)$ | ⬜ | |

### 2. Ensemble Methods
*Methods for combining weak learners to reduce bias and variance.*

| Topic | Key Math / Concept | Status | Implementation |
| :--- | :--- | :---: | :--- |
| **Decision Trees** | Gini Impurity / Information Gain | ✅ | `tree/decision_trees.py` |
| **Bagging** | Variance reduction / Bootstrapping | ✅ | `ensembles/bagging.py` |
| **AdaBoost** | Weighted Error Minimization | ✅ | `ensembles/adaboost.py` |
| **Gradient Boosting**| Residual fitting via Gradient Descent | ✅ | `ensembles/gbm.py` |
| **XGBoost** | 2nd-order Taylor expansion | ⬜ | |
| **LightGBM** | Histogram-based growth | ⬜ | |
| **CatBoost** | Ordered Boosting | ⬜ | |

### 3. Support Vector Machines (SVM)
*Maximum margin classifiers and kernel tricks.*

| Topic | Key Math / Concept | Status | Implementation |
| :--- | :--- | :---: | :--- |
| **Hard/Soft Margin** | Hinge Loss & Slacks ($\xi$) | ⬜ | |
| **Linear Kernel** | Hyperplane: $w^T x + b$ | ⬜ | |
| **Polynomial Kernel** | Manual Feature Expansion | ⬜ | |
| **RBF/Gaussian** | Infinite Dimensional Mapping | ⬜ | |

### 4. Probabilistic & Generative Models
*Bayesian approaches and density estimation.*

| Topic | Key Math / Concept | Status | Implementation |
| :--- | :--- | :---: | :--- |
| **Multivariate Gaussian**| Covariance Matrix $\Sigma$ | ⬜ | |
| **Gaussian Processes**| Kernel/Covariance Functions | ⬜ | |
| **GMM** | Expectation-Maximization (EM) | ⬜ | |
| **LDA vs QDA** | Decision Boundary Geometry | ⬜ | |

### 5. Unsupervised Learning & Dimensionality Reduction
*Finding structure in unlabeled data.*

| Topic | Key Math / Concept | Status | Implementation |
| :--- | :--- | :---: | :--- |
| **k-means** | Centroid minimization | ⬜ | |
| **Hierarchical** | Linkage types (Ward/Complete) | ⬜ | |
| **PCA** | Eigenvalues / Variance Maximization | ⬜ | |
| **t-SNE** | KL-Divergence / Manifold mapping | ⬜ | |
| **Autoencoders** | Bottleneck Reconstruction Loss | ✅ | `nn/models/feed_forward.py` |
