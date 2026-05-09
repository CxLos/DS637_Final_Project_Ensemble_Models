# Final Project - Ensemble Models: Plain English Breakdown

---

## What Is This Project About?

You are building a **wine quality predictor**. Given 10 measurements about a wine (like its acidity, sugar, alcohol level, etc.), you want to predict whether the wine is good or bad (binary classification).

The twist: the professor does not want you to just call `sklearn.linear_model.LogisticRegression()` and call it a day. He wants you to:
1. Build logistic regression **from scratch** using math and `numpy`
2. Then build a **tree of logistic regression models** (ensemble) that should be more accurate
3. Then go even deeper with a **3-layer tree**

---

## The Dataset

File: `allwine.csv`

**Input features (columns):** 
  - fixed acidity, 
  - volatile acidity, 
  - citric acid, 
  - residual sugar,
  - chlorides,
  - free sulfur dioxide, density,
  - pH, 
  - sulphates, 
  - alcohol

**Target (what you are predicting):** `quality` -- this is the column you want to get right

**Note:** The features are already pre-scaled (StandardScaler applied). `quality` is binary: **0 = bad wine, 1 = good wine**.

#### What Does "Already Scaled" Mean?

`StandardScaler` transforms each feature so it has **mean = 0** and **standard deviation = 1** using:

$$z = \frac{x - \mu}{\sigma}$$

- Subtract the average (`mu`) from each value, then divide by the spread (`sigma`)
- The result is measured in **"standard deviations from the mean"** -- no hard cap
- Most values fall roughly in the **-3 to +3 range** (that's just how normal distributions work -- ~99.7% of data lands within ±3 std devs)
- `0.0` = exactly average for that feature
- `1.5` = 1.5 standard deviations above average
- `-2.0` = 2 standard deviations below average

**This is NOT the same as MinMaxScaler**, which caps values between 0 and 1. StandardScaler has no hard cap -- values like `-1.455` or `1.798` are completely normal.

**Why it matters here:** Since `allwine-1.csv` is already scaled, applying `StandardScaler` again in your code won't break anything (it'll just come out nearly identical), but it's technically redundant.

**Dataset shape:** 3198 rows x 11 columns (1488 bad, 1710 good)

### Sample of the data:

```
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide   density        pH  sulphates   alcohol  quality
0      -0.743787          0.805266    -1.455948       -0.541531  -0.334525            -0.539436 -0.978159  0.146723  -0.755850 -1.297136        0
1      -0.520914          1.798500    -1.455948       -0.047918   0.129345             0.787432 -0.998211 -1.220838  -0.062351 -0.960761        0
2      -0.520914          1.136344    -1.251203       -0.259467   0.002835            -0.160331 -0.994200 -0.956148  -0.235726 -0.960761        0
3       1.373509         -1.512280     1.410480       -0.541531  -0.355610             0.029222 -0.974148 -1.397297  -0.640267 -0.960761        1
4      -0.743787          0.805266    -1.455948       -0.541531  -0.334525            -0.539436 -0.978159  0.146723  -0.755850 -1.297136        0
```

---

## Allowed Libraries ONLY

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

> **DO NOT** use `sklearn.linear_model.LogisticRegression` or any other pre-built model. You are writing the math yourself.

---

## Background: What Is Logistic Regression?

Logistic regression is a way to predict a **yes/no (0 or 1) answer** from a set of numbers (your features).

It uses a special math function called the **sigmoid function** (also called the logistic function):

$$p(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x)}}$$

**In plain English:**
- You take your 10 wine measurements and multiply each one by a "weight" (called theta, th)
- You add them all together: `th_0 + th_1*x1 + th_2*x2 + ... + th_10*x10`
- You plug that sum into the sigmoid formula above
- The output is a number between 0 and 1 -- that is your **predicted probability** that the wine is good
- If probability > 0.5, predict "good wine" (1). Otherwise predict "bad wine" (0)

**Your thetas (th) are what the model "learns".** Training the model means finding the best theta values that make the most correct predictions.

---

## Background: How Do You Train It? (Maximum Likelihood Estimation)

You adjust the theta values to maximize the **likelihood function**:

$$L(\theta) = \prod_{k: y_k=1} p_k \cdot \prod_{k: y_k=0} (1 - p_k)$$

**In plain English:**
- For every wine that IS actually good (y=1), you want your model's predicted probability `p` to be as HIGH as possible
- For every wine that is NOT good (y=0), you want `(1 - p)` to be as HIGH as possible (i.e., `p` should be low)
- Multiply all those probabilities together -- that product is your "likelihood"
- The better your theta values, the higher this number gets

**In practice, you maximize the log-likelihood** (easier math, same result):

$$\ell = \sum_{k=1}^{K} \left[ y_k \ln(p_k) + (1 - y_k) \ln(1 - p_k) \right]$$

> **Tip:** Follow the linked GitHub sample notebook step by step (In[1] through In[29]). You can reuse that code directly.
> Link: https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb

---

# TASK 1 (1 pt) -- Build a Basic Logistic Regression From Scratch

## What You Need To Do:

1. **Load and clean the data** -- use `allwine.csv`
2. **Split into features and target** -- X = the 10 wine measurement columns, y = `quality`
3. **Split into train/test sets** -- use `train_test_split`
4. **Scale the features** -- use `StandardScaler` so all features are on the same scale
5. **Implement the sigmoid function** from scratch using `numpy`
6. **Implement the log-likelihood cost function** from scratch
7. **Train the model** by finding the best theta values (use gradient ascent or the optimizer from the sample notebook)
8. **Test the model** -- use `accuracy_score` to see how accurate your predictions are on the test set
9. **Report the accuracy** -- this is your baseline. Task 2 should beat it.

## Key Code Structure:

```python
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction
def predict(X, theta):
    return sigmoid(X @ theta)  # matrix multiply features by weights

# Log-likelihood
def log_likelihood(X, y, theta):
    p = predict(X, theta)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
```
---

# TASK 2 (4 pts) -- Build a 2-Layer Ensemble (Tree of Logistic Regressions)

## The Big Idea:

Instead of ONE logistic regression model, you now build a **small tree with 3 models**:

```
          LR_middle  (top layer -- the "router")
         /          \
      LR_left      LR_right   (second layer -- the "classifiers")
```

- **`LR_middle`** -- looks at the wine data and decides: "send this sample LEFT or RIGHT?" It does not make a final prediction itself; it just splits the data.
- **`LR_left`** -- makes the final yes/no quality prediction for samples routed left
- **`LR_right`** -- makes the final yes/no quality prediction for samples routed right

Think of it like a "router" at the top that splits the data into two groups, and then each group gets its own specialist logistic regression model.

## The Math (Combined Probability):

For a given wine sample x, the combined probability of predicting y=1 (good wine) is:

$$P(y=1 | x) = h_L(x) \cdot h_M(x) + h_R(x) \cdot (1 - h_M(x))$$

**In plain English:**
- `h_M(x)` = probability that `LR_middle` routes the sample LEFT
- `(1 - h_M(x))` = probability `LR_middle` routes it RIGHT
- `h_L(x)` = `LR_left`'s prediction (probability of good wine)
- `h_R(x)` = `LR_right`'s prediction (probability of good wine)
- The final probability is a **weighted average** of LR_left and LR_right, where the weights come from LR_middle

And the probability of y=0 (bad wine):

$$P(y=0 | x) = (1 - h_L(x)) \cdot h_M(x) + (1 - h_R(x)) \cdot (1 - h_M(x))$$

You can verify: P(y=1) + P(y=0) = 1

## The Likelihood Function To Maximize:

$$L(\theta_M, \theta_L, \theta_R) = \prod_{i=1}^{n} P(y_i | x_i, \theta_M, \theta_L, \theta_R)$$

**In plain English:**
- You have **30 total parameters** to learn:
  - `theta_M` = 10 weights for LR_middle
  - `theta_L` = 10 weights for LR_left
  - `theta_R` = 10 weights for LR_right
- You adjust all 30 values at once to maximize the likelihood above
- **DO NOT assume the same weights** for all 3 nodes -- each model has its own independent set of thetas

## Important Notes From The Professor:

> "Please do NOT assume the 'same weights' from 3 nodes because there is no place in the project mentioning it."
> "You have a function L with 30 parameters. Your goal is to find the 30 values for your 30 thetas in order to maximize the above function."

## What You Need To Do:

1. Initialize `theta_M`, `theta_L`, `theta_R` separately (e.g., as zeros or random small values)
2. Write a function that computes P(y=1|x) using the formula above
3. Write the log-likelihood using ln(P(y|x)) summed over all samples
4. Use gradient ascent (or `scipy.optimize.minimize` with `method='BFGS'`) to update all 30 thetas to maximize log-likelihood
5. Predict on the test set -- your accuracy should be **better than Task 1**

---

# TASK 3 (5 pts) -- Build a 3-Layer Deep Ensemble Model

## The Big Idea:

Extend the tree to **3 layers** with **7 total nodes** (like a full binary tree):

```
Layer 1:               Node_1              (1 model -- router)
                      /      \
Layer 2:         Node_2      Node_3        (2 models -- routers)
                /    \      /    \
Layer 3:    Node_4  Node_5 Node_6  Node_7  (4 models -- leaf predictors)
```

- **Layers 1 and 2** (nodes 1, 2, 3) are "routers" -- logistic regression models that split probability to their two children
- **Layer 3** (nodes 4, 5, 6, 7) are "leaves" -- these make the actual final predictions
- The **final prediction** is a **probability-weighted sum** of all 4 leaf predictions

Think of it as: the top node routes the data, each second-layer node routes further, and the bottom nodes each make a prediction. The final answer is a blend of all 4 leaf predictions, weighted by how much probability flowed to each leaf.

## What You Need To Do:

1. **Derive the likelihood function** for this 3-layer tree (follow the same logic as Task 2 but extended one more level)
2. You will have **7 sets of thetas** (7 models x 10 features = 70 parameters total)
3. Implement the combined probability formula for the 3-layer tree
4. Train the model by maximizing the log-likelihood across all 70 parameters
5. Test performance -- should beat Task 2

## How To Think About The Math:

In Task 2, P(y=1) was:

```
P(y=1) = h_L * h_M + h_R * (1 - h_M)
```

In Task 3, each of h_L and h_R themselves become routers to two children. The weights that flow to each leaf are:

- Node_4 (left leaf of Node_2):  weight = h_1 * h_2
- Node_5 (right leaf of Node_2): weight = h_1 * (1 - h_2)
- Node_6 (left leaf of Node_3):  weight = (1 - h_1) * h_3
- Node_7 (right leaf of Node_3): weight = (1 - h_1) * (1 - h_3)

Final prediction:

```
P(y=1) = h_4 * (h_1 * h_2)
        + h_5 * (h_1 * (1 - h_2))
        + h_6 * ((1 - h_1) * h_3)
        + h_7 * ((1 - h_1) * (1 - h_3))
```

Where h_4, h_5, h_6, h_7 are the leaf models' sigmoid outputs (final predictions).

---

# TASK 4 (Bonus, 0 pts) -- Generalize to N Layers

## The Big Idea:

Instead of hardcoding 2 or 3 layers, write code that builds a tree of **any depth** automatically.

- This is essentially a recursive version of Task 2 and Task 3
- If you complete this, **your lowest quiz score gets dropped** (quiz weight goes from avg of 4 quizzes to avg of 3 best quizzes, both divided by 5)

---

# Grading Summary

| Task   | Points | Description                          |
|--------|--------|--------------------------------------|
| Task 1 | 1 pt   | Logistic regression from scratch     |
| Task 2 | 4 pts  | 2-layer ensemble (3 LR models)       |
| Task 3 | 5 pts  | 3-layer ensemble (7 LR models)       |
| Task 4 | Bonus  | N-layer generalization (drops lowest quiz) |
| Solo   | +2 pts | Working alone gives 2 bonus points   |
| 2-person | +1 pt each | 2-person team gets 1 bonus point each |

---

# Submission Requirements

- Submit both: `.ipynb` file AND `.html` file with all outputs visible
- Each group submits one copy on Canvas
- Include all team member names at the top of the notebook
- **Late policy:** -2.5 points per late day

---

# Recommended Workflow

1. Start with Task 1 -- get logistic regression working from the sample notebook
2. Record the accuracy from Task 1 (you need to beat it in Task 2)
3. For Task 2, write three sigmoid/predict functions and combine them in the P(y|x) formula
4. Use `scipy.optimize.minimize` with `method='BFGS'` to optimize -- it handles multi-parameter optimization cleanly
5. For Task 3, extend the probability formula one more layer using the tree weights above
6. Convert notebook to HTML: File > Export > HTML, or run `jupyter nbconvert --to html your_notebook.ipynb`
