## 1. Major Data Pitfalls (From the Slides)
The primary pitfalls identified in your lecture materials focus on the quality and integrity of your training data:

- **Sampling Bias**: Occurs when the training data is not representative of the real-world environment where the model will be deployed. For example, a housing model trained only on data from sunny coastal cities will fail to predict prices accurately in mountain regions.

- **Irrelevant Feature Selection**: Including "noisy" or unimportant variables that have no relationship with the target. This can confuse the model and lead it to find patterns where none exist.

- **Missing Data**: Null or empty values in your dataset. If not handled through imputation or removal, these can cause algorithms to crash or produce biased results.

- **Inaccurate Scaling and Normalization**: This happens when features have vastly different units (e.g., kilograms vs. grams). Without proper scaling (like StandardScaler), the model may prioritize a feature simply because its numbers are larger, rather than because it is more important.

- **Neglecting Outliers**: Failing to identify extreme values that don't follow the general trend. As seen in your Residual Plot, a single massive outlier can significantly inflate your Mean Squared Error (MSE).

- **Miscalculated Features**: Logical errors during the feature engineering phase. If the "math" behind a feature is wrong, the model’s entire foundation is flawed.

## 2. Additional "Industry Standard" Pitfalls
While your slides list six, the authentic source (VentureBeat) identifies nine core pitfalls to be "exhaustive":

- **Data Leakage**: This is perhaps the most common pitfall for practitioners. It occurs when information from the "future" (the test set) is inadvertently allowed to influence the training set. A prime example is using fit_transform on your entire dataset before splitting it into train and test sets.

- **Overfitting**: Creating a model that is so complex it "memorizes" the training data but fails to generalize to new, unseen data. This is often detected when your training MSE is much lower than your cross-validation MSE.

- **Lack of Interpretability**: Building "Black Box" models that provide a prediction without explaining why. In critical fields like your NLP for depression analysis or legal applications, explainability (XAI) is vital for trust and safety.

## 3. Practical Pitfalls Observed in Your Code
Based on your recent debugging sessions, we can add three "implementation-level" pitfalls to watch for in your upcoming exam or project:

- **Model Mismatch**: Using the wrong type of algorithm for the task (e.g., using LogisticRegression for a price prediction task instead of LinearRegression).

- **Unpacking Order Errors**: Swapping the variables in train_test_split (e.g., X_train, Y_train, X_test, Y_test), which leads to training your model on the features instead of the labels.

- **The NotFittedError**: Attempting to make a prediction (.predict()) before actually training the model on the data (.fit()).

---
## The MSE Gap
The "training MSE vs. cross-validation MSE" comparison is the definitive way to diagnose this.

Training MSE: This measures how well the model fits the data it was trained on. A very low value (near zero) means the model has perfectly "memorized" these points.

Cross-Validation (CV) MSE: This measures how the model performs on small subsets of data it didn't see during that specific training fold.

The Overfitting Gap: If your Training MSE is 0.05 but your CV MSE is 0.5268, the model is overfitting. It is performing 10x worse on unseen data, proving it hasn't generalized well.

## Why Models Overfit
Based on your lecture materials and common pitfalls, here are the main triggers:

Excessive Complexity: Using a high-degree polynomial or a very deep Neural Network for a simple linear relationship.

Small Dataset: If you only have a few labeled examples (like your 10 labeled emails), the model can easily find "fake" patterns that don't exist in the larger population.

Irrelevant Features: Including "noise" features (like the day of the week a house was sold) can lead the model to believe those factors drive the price.

## How to Fix It
Regularization (L1/L2): As mentioned in your exam prep questions, adding a "penalty" to the model's weights forces it to stay simple.

Cross-Validation: Using cross_val_score ensures you catch overfitting early by testing the model on different "folds" of the data.

Pruning: In Tree-based models (like the Random Forest we discussed), you can limit the "depth" of the trees so they don't grow too complex.

---

1. **L1 Regularization (Lasso)**
- The "L" in Lasso stands for Least Absolute Shrinkage and Selection Operator. It adds a penalty equal to the absolute value of the magnitude of coefficients.
- Penalty Term: $\lambda \sum |w_i|$ 
- Key Property (Sparsity): L1 has a unique ability to drive the weights of unimportant features to exactly zero.Best Use Case: When you have many features and you suspect only a few are actually important. It effectively performs Feature Selection for you.Example: You are predicting house prices with 100 features, including "color of the front door." L1 will likely set the weight for door color to 0, removing it from the model entirely.
2. **L2 Regularization (Ridge)**
- Ridge regression adds a penalty equal to the square of the magnitude of coefficients.
- Penalty Term: $\lambda \sum w_i^2$
- Key Property (Shrinkage): L2 shrinks the weights but almost never makes them zero. It keeps all features but minimizes their impact.
- Best Use Case: When you have many features that are interdependent or correlated, and you believe most of them contribute at least a little to the result.
- Example: Using your California Housing data. Features like "Median Income" and "Number of Rooms" are related. L2 will keep both but ensure neither has a "massive" weight that overpowers the other.

3. **The $\lambda$ (Lambda) Factor**
- In both cases, you will see a parameter called $\lambda$ (or alpha in Scikit-Learn).
- Small $\lambda$: The penalty is weak. The model behaves like a standard Linear Regression (high risk of overfitting).
- Large $\lambda$: The penalty is strong. The model is forced to be very simple (risk of underfitting).
- How to implement in your code?In Scikit-Learn, you don't usually use LinearRegression with a manual penalty. Instead, you use the dedicated classes:
```
from sklearn.linear_model import Lasso, Ridge

# For L1 (Lasso)
lasso_reg = Lasso(alpha=1.0) # alpha is your lambda
lasso_reg.fit(X_train, Y_train)

# For L2 (Ridge)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, Y_train)
```

## Imputation
- Imputation is a statistical and data preprocessing technique used to replace missing data with substituted values.
- **Why Impute Data?**
    - In your lecture slides, Missing Data is listed as Pitfall 3 in building a successful ML program. If you simply delete rows with missing values (a process called "Listwise Deletion"), you risk:
        - Losing Valuable Information: You might throw away 50% of your dataset just because one column has a few missing spots.
        - Introducing Bias: If data is missing for a specific reason (e.g., lower-income households being less likely to report income), deleting those rows makes your model unrepresentative of that group.

```
from sklearn.impute import SimpleImputer
import numpy as np

# 1. Initialize the imputer (using median for housing data)
imputer = SimpleImputer(strategy='median')

# 2. Fit and transform your features
# Remember: Fit on Train, only Transform on Test to avoid leakage!
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

## 