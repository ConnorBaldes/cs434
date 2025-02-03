# CS434: Machine Learning and Data Mining

CS434 provides a broad introduction to **machine learning** and **data mining**, covering both **computational** and **statistical** perspectives. This course explores **supervised and unsupervised learning**, **ensemble methods**, and **model evaluation** techniques, with a focus on practical application and theoretical understanding. The projects in this repository reinforce key concepts through hands-on implementation of essential machine learning algorithms.

## Key Topics
- **Supervised Learning**: Linear models, support vector machines (SVMs), neural networks, Naïve Bayes.
- **Unsupervised Learning**: Clustering (K-Means), dimensionality reduction.
- **Ensemble Learning**: Bagging, boosting, random forests.
- **Model Selection & Evaluation**: Cross-validation, bias-variance tradeoff, performance metrics.
- **Probability & Statistics**: Maximum likelihood estimation, Bayesian inference.
- **Optimization & Computational Complexity**: Gradient descent, kernel methods, perceptron.

## Course Objectives
By completing this course, one will:
- **Formulate machine learning problems** for various applications.
- **Analyze and implement fundamental machine learning algorithms**.
- **Evaluate models** using statistical and computational methods.
- **Understand algorithmic strengths, weaknesses, and trade-offs**.
- **Apply machine learning techniques** to solve real-world problems.

## Repository Structure
The repository is organized into projects focusing on different machine learning techniques:
```
cs434/ 
│── README.md 
│── .git/ 
└── projects/ 
    ├── decision_trees_and_k_means_clustering/ 
    ├── knn_classification_and_statistical_estimation/ 
    ├── linear_models_for_regression_and_classification/ 
    ├── naive_bayes_and_neural_networks/
```

## Projects

### [**K-Nearest Neighbor Classification & Statistical Estimation**](./projects/knn_classification_and_statistical_estimation/)
Located in: `projects/knn_classification_and_statistical_estimation/`

This project introduces **k-Nearest Neighbor (kNN) classification** and **statistical estimation** techniques, both foundational in **supervised learning** and **probabilistic modeling**. The project provides hands-on experience implementing a **non-parametric classification algorithm** and conducting **maximum likelihood estimation (MLE)** and **Bayesian inference**.

#### **Key Concepts**
| Concept                     | Description |
|-----------------------------|-------------|
| **k-Nearest Neighbors (kNN)** | A simple, instance-based classification algorithm that assigns labels based on the majority class of the k closest training points. |
| **Distance Metrics**        | Computing similarity using Euclidean distance and its relation to vector norms. |
| **Hyperparameter Tuning**   | Optimizing kNN by selecting the best `k` using cross-validation. |
| **K-Fold Cross Validation** | Partitioning the dataset to evaluate model generalization and mitigate overfitting. |
| **Maximum Likelihood Estimation (MLE)** | Estimating the Poisson distribution parameter λ using likelihood functions. |
| **Bayesian Inference**      | Using the Gamma distribution as a conjugate prior to the Poisson model for parameter estimation. |

#### **Project Breakdown**
##### **1. Statistical Estimation**
- Derive and compute the **maximum likelihood estimate (MLE)** of the Poisson distribution parameter **λ**.
- Implement **Maximum A Posteriori (MAP) estimation** using a **Gamma prior**.
- Demonstrate that the Gamma distribution is a **conjugate prior** to the Poisson.

##### **2. k-Nearest Neighbors (kNN) Implementation**
- Implement **kNN classification** from scratch in Python using **NumPy** (no external ML libraries allowed).
- Compute **Euclidean distance** efficiently using **vectorized operations**.
- Predict class labels for income classification based on census data.

##### **3. Model Selection & Cross Validation**
- Implement **4-fold cross-validation** to assess model performance.
- Evaluate accuracy trends for different `k` values (`1, 3, 5, 7, 9, 99, 999, 8000`).
- Analyze results to identify underfitting and overfitting behaviors.

##### **4. Kaggle Competition**
- Optimize kNN for a **real-world dataset** and submit predictions to a **class leaderboard**.
- Experiment with feature selection, distance weighting, and hyperparameter tuning.

#### **Course Concepts Covered:**

- **Supervised Learning**: Applying **instance-based** classification.
- **Statistical Learning**: Deriving **MLE** and Bayesian estimates.
- **Model Selection**: Implementing **cross-validation** to optimize performance.
- **Computational Complexity**: Efficient **vectorized distance calculations**.

---

### [**Linear Models for Regression & Classification**](./projects/linear_models_for_regression_and_classification/)
Located in: `projects/linear_models_for_regression_and_classification/`

This project introduces **linear regression** and **logistic regression**, two fundamental **supervised learning** techniques used for **prediction and classification**. The project emphasizes probabilistic modeling, optimization, and model evaluation.

#### **Key Concepts**
| Concept                         | Description |
|---------------------------------|-------------|
| **Linear Regression**            | Predicts continuous values by fitting a linear model to data. |
| **Loss Functions**               | Comparison of Sum of Squared Errors (SSE) vs. Sum of Absolute Errors (SAE). |
| **Logistic Regression**          | Binary classification model using a sigmoid activation function. |
| **Negative Log-Likelihood (NLL)**| Objective function minimized during logistic regression training. |
| **Gradient Descent Optimization**| Iteratively updates weights to minimize loss. |
| **Regularization**               | Controlling model complexity to prevent overfitting. |
| **Precision & Recall**           | Evaluating classification performance beyond accuracy. |
| **K-Fold Cross Validation**      | Assessing model generalization and hyperparameter tuning. |

#### **Project Breakdown**
##### **1. Statistical Foundations of Regression**
- Derive the **MLE** for **linear regression** under different probabilistic assumptions (Gaussian vs. Laplace noise).
- Compare **Sum of Squared Errors (SSE)** and **Sum of Absolute Errors (SAE)**.
- Compute **recall and precision** at various probability thresholds to understand classification performance.

##### **2. Implementing Logistic Regression for Tumor Classification**
- Implement **logistic regression** to classify tumors as **benign or malignant**.
- Optimize model parameters using **gradient descent**.
- Evaluate **training loss trends** to analyze convergence.

##### **3. Model Tuning & Evaluation**
- Implement **gradient-based optimization** for logistic regression.
- Explore the impact of **learning rate selection** on model convergence.
- Introduce a **bias term** and observe its effect on model accuracy.
- Use **cross-validation (K=2,3,4,5,10,20,50)** to assess generalization.

##### **4. Kaggle Competition**
- Train a logistic regression model on a **real-world dataset** and submit predictions to a **class leaderboard**.
- Experiment with **feature transformations, learning rates, and hyperparameter tuning**.

#### **Course Concepts Covered:**
- **Supervised Learning**: Applying **linear** and **logistic regression** for prediction and classification.
- **Statistical Learning**: Exploring **MLE-based estimation**.
- **Optimization**: Implementing **gradient descent** for model training.
- **Model Selection**: Using **cross-validation** for hyperparameter tuning.
- **Performance Evaluation**: Understanding **precision-recall trade-offs**.

---

### [**Naïve Bayes & Neural Networks**](./projects/naive_bayes_and_neural_networks/)
Located in: `projects/naive_bayes_and_neural_networks/`

This project introduces **Naïve Bayes classification** and **feedforward neural networks**, two contrasting approaches to classification. The project covers **probabilistic modeling**, **Bayesian inference**, and **deep learning fundamentals**, reinforcing key principles of **supervised learning** and **optimization**.

#### **Key Concepts**
| Concept                         | Description |
|---------------------------------|-------------|
| **Naïve Bayes Classification**  | Probabilistic classifier assuming feature independence given the class. |
| **Bayesian Inference**          | Computing class probabilities using prior and likelihood estimates. |
| **Bernoulli Naïve Bayes**       | Special case where features are binary, leading to a linear decision boundary. |
| **Neural Networks**             | Learning complex decision boundaries using multi-layer perceptrons. |
| **Backpropagation**             | Efficient gradient computation for neural network training. |
| **Cross-Entropy Loss**          | Optimizing multi-class classification models. |
| **Gradient Descent Optimization** | Updating model parameters via iterative minimization of loss functions. |
| **Hyperparameter Tuning**       | Adjusting model architecture and training settings for optimal performance. |

#### **Project Breakdown**
##### **1. Naïve Bayes Classifier**
- Implement a **Bernoulli Naïve Bayes classifier** for binary feature datasets.
- Derive the **linear decision boundary** formed by Naïve Bayes under certain assumptions.
- Analyze the impact of **duplicated features** on classifier confidence.

##### **2. Implementing a Neural Network for Digit Classification**
- Train a **multi-layer feedforward neural network** to classify digits from a **subset of the MNIST dataset**.
- Use **cross-entropy loss** for multi-class classification.
- Apply **gradient descent** to optimize neural network weights.

##### **3. Backpropagation & Optimization**
- Implement **backpropagation** to compute gradients for weight updates.
- Optimize model performance by adjusting:
  - **Learning rate (step size)**
  - **Number of layers & hidden units**
  - **Activation functions (ReLU vs. Sigmoid)**

##### **4. Model Evaluation & Hyperparameter Tuning**
- Experiment with different **learning rates** and **network depths** to observe training behavior.
- Analyze the **vanishing gradient problem** in deep networks.
- Evaluate **random seed sensitivity** to understand the impact of initialization.

##### **5. Kaggle Competition**
- Train a **digit recognition model** and submit predictions to a **class leaderboard**.
- Optimize network hyperparameters to improve classification accuracy.

#### **Course Concepts Covered:**
- **Supervised Learning**: Implementing **probabilistic classifiers** and **neural networks**.
- **Bayesian Modeling**: Applying **prior probabilities** for classification.
- **Deep Learning Foundations**: Training **multi-layer perceptrons (MLPs)**.
- **Optimization**: Using **gradient descent and backpropagation**.
- **Hyperparameter Tuning**: Adjusting **learning rates, activations, and depth**.

---

### [**Decision Trees & k-Means Clustering**](./projects/decision_trees_and_k_means_clustering/)
Located in: `projects/decision_trees_and_k_means_clustering/`

This project explores **decision tree learning**, a key method for supervised classification, and **k-Means clustering**, a widely used unsupervised learning algorithm. The project covers **model interpretability, feature selection, ensemble learning, and clustering techniques**, reinforcing core principles of **CS434**.

#### **Key Concepts**
| Concept                         | Description |
|---------------------------------|-------------|
| **Decision Trees**               | Recursive partitioning method for classification. |
| **Information Gain & Entropy**   | Evaluating feature importance in tree construction. |
| **Overfitting & Pruning**        | Techniques to improve generalization. |
| **Random Forests & Bagging**     | Reducing model variance by ensembling decision trees. |
| **k-Means Clustering**           | Partitioning data into k clusters based on similarity. |
| **Centroid Initialization**      | Exploring different methods for selecting initial cluster centers. |
| **Sum of Squared Errors (SSE)**  | Measuring clustering quality. |
| **Feature Representations**      | Using Histogram of Oriented Gradients (HOG) for image clustering. |

#### **Project Breakdown**
##### **1. Decision Tree Learning**
- **Understanding Decision Boundaries**: Visualizing how decision trees partition input space.
- **Information Gain & Feature Selection**: Learning how decision trees determine optimal splits.
- **Constructing a Decision Tree**: Implementing a tree classifier from training data.
- **Random Forests & Ensemble Methods**: Measuring **correlation** between trees and improving performance through **bagging**.

##### **2. Implementing k-Means Clustering**
- **Centroid Initialization**: Comparing random selection, k-Means++, and other methods.
- **Cluster Assignments**: Assigning each data point to the nearest centroid.
- **Centroid Updates**: Computing cluster means and iterating until convergence.
- **Evaluating Clustering Stability**: Running k-Means multiple times to observe variations.

##### **3. Choosing the Right Number of Clusters**
- **Elbow Method & SSE**: Plotting SSE vs. `k` to determine an optimal cluster count.
- **Hyperparameter Sensitivity**: Investigating how initialization and parameter choices impact results.

##### **4. Clustering Images with HOG Features**
- **Extracting HOG Features**: Using edge detection for structured feature representation.
- **Applying k-Means to Images**: Grouping visually similar images into clusters.
- **Cluster Purity Analysis**: Assigning semantic labels to clusters and evaluating classification accuracy.

##### **5. Kaggle Competition**
- Train an **unsupervised clustering model** and submit results to a **class leaderboard**.
- Optimize **feature extraction and cluster initialization** for improved performance.

#### **Concepts Covered in the Course**
This project aligns with **CS434 course objectives**, reinforcing:
- **Supervised Learning**: Implementing and analyzing **decision trees**.
- **Unsupervised Learning**: Understanding **k-Means clustering** and **feature extraction**.
- **Model Selection**: Applying **cross-validation for tree pruning** and **SSE for clustering evaluation**.
- **Ensemble Learning**: Using **random forests and bagging** to improve decision trees.
- **Feature Engineering**: Extracting **HOG features** for image-based clustering.

---

