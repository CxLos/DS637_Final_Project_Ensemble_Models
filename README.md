# Final Project - Ensemble Models 📈

---

## What Is This Project About?

You are building a **wine quality predictor**. Given 10 measurements about a wine (like its acidity, sugar, alcohol level, etc.), you want to predict whether the wine is good or bad (binary classification).

The twist: the professor does not want you to just call `sklearn.linear_model.LogisticRegression()` and call it a day. He wants you to:
1. Build logistic regression **from scratch** using math and `numpy`
2. Then build a **tree of logistic regression models** (ensemble) that should be more accurate
3. Then go even deeper with a **3-layer tree**

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
  - quality

**Target (what you are predicting):** `quality` -- this is the column you want to get right

**Note:** The features are already pre-scaled (StandardScaler applied). `quality` is binary: **0 = bad wine, 1 = good wine**.

### `df.head(10):`

```
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide   density        pH  sulphates   alcohol  quality
0      -0.743787          0.805266    -1.455948       -0.541531  -0.334525            -0.539436 -0.978159  0.146723  -0.755850 -1.297136        0
1      -0.520914          1.798500    -1.455948       -0.047918   0.129345             0.787432 -0.998211 -1.220838  -0.062351 -0.960761        0
2      -0.520914          1.136344    -1.251203       -0.259467   0.002835            -0.160331 -0.994200 -0.956148  -0.235726 -0.960761        0
3       1.373509         -1.512280     1.410480       -0.541531  -0.355610             0.029222 -0.974148 -1.397297  -0.640267 -0.960761        1
4      -0.743787          0.805266    -1.455948       -0.541531  -0.334525            -0.539436 -0.978159  0.146723  -0.755850 -1.297136        0
5      -0.743787          0.584547    -1.455948       -0.612047  -0.355610            -0.349884 -0.978159  0.146723  -0.755850 -1.297136        0
6      -0.465196          0.253469    -1.148831       -0.753079  -0.482120            -0.160331 -1.006232 -0.779689  -1.333765 -1.297136        0
7      -0.799505          0.529368    -1.455948       -1.035144  -0.566460            -0.160331 -1.042326 -0.382655  -1.275973 -0.792573        1
8      -0.520914          0.143110    -1.353576       -0.471015  -0.397780            -0.728989 -0.998211 -0.515000  -0.698058 -1.213043        1
9      -0.688069         -0.298327     0.386756        2.420145  -0.439950             0.029222 -0.978159 -0.559115   0.631147 -0.372104        0
```

### `df.info()`

```
<class 'pandas.core.frame.DataFrame'>
Index: 3198 entries, 0 to 3197
Data columns (total 11 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   fixed acidity        3198 non-null   float64
 1   volatile acidity     3198 non-null   float64
 2   citric acid          3198 non-null   float64
 3   residual sugar       3198 non-null   float64
 4   chlorides            3198 non-null   float64
 5   free sulfur dioxide  3198 non-null   float64
 6   density              3198 non-null   float64
 7   pH                   3198 non-null   float64
 8   sulphates            3198 non-null   float64
 9   alcohol              3198 non-null   float64
 10  quality              3198 non-null   int64  
dtypes: float64(10), int64(1)
memory usage: 299.8 KB
```

**Key takeaways:**
- **No missing values** — all 3198 rows are complete across all 11 columns, no imputation needed
- **10 float features** — already scaled (StandardScaler applied), so all values are z-scores
- **1 integer target** — `quality` is binary: 0 = bad wine, 1 = good wine
- **No nulls to handle** — you can go straight to train/test split

### `df.describe()`

```
       fixed acidity  volatile acidity   citric acid  residual sugar     chlorides  free sulfur dioxide       density            pH     sulphates       alcohol      quality
count   3198.000000       3198.000000    3198.000000     3198.000000    3198.000000          3198.000000   3198.000000   3198.000000   3198.000000   3198.000000  3198.000000
mean       ~0.000000         ~0.000000      ~0.000000       ~0.000000      ~0.000000            ~0.000000     ~0.000000     ~0.000000     ~0.000000     ~0.000000     0.534709
std        1.000156          1.000156       1.000156        1.000156       1.000156             1.000156      1.000156      1.000156      1.000156      1.000156      0.498872
min       -2.303900         -2.395155      -1.623825       -1.246692      -1.683967            -1.487200     -1.133162     -3.250120     -2.085055     -2.138075      0.000000
25%       -0.688069         -0.741392      -0.904120       -0.471015      -0.376695            -0.793092     -0.999214     -0.779689     -0.640267     -0.713777      0.000000
50%       -0.188304         -0.037766      -0.066512       -0.259467      -0.174435            -0.214011     -0.000717     -0.002683     -0.197714     -0.126251      1.000000
75%        0.563855          0.622155       0.796246        0.067057       0.085979             0.548119      0.999093      0.756477      0.433853      0.638319      1.000000
max        4.580738          5.932391       3.824068        9.138676      11.115258             5.245422      1.140202      3.769595      7.944553      4.311531      1.000000
```

**Key takeaways:**
- **mean ≈ 0.0 for all features** — confirms StandardScaler was applied (the tiny non-zero values like `-3.7e-16` are floating-point rounding noise, effectively 0)
- **std ≈ 1.0 for all features** — confirms each feature has been normalized to unit variance
- **quality mean = 0.5347** — the dataset is nearly balanced: ~53.5% good wine (1), ~46.5% bad wine (0)
- **quality median = 1.0** — more than half the samples are good wine (50th percentile is 1)
- **Outliers exist** — `chlorides` max = 11.12 std devs above mean, `volatile acidity` max = 5.93 std devs. These are extreme but valid scaled values (the original raw data had outliers)
- **No feature has max = 1.0** — confirms this is StandardScaler, not MinMaxScaler

#### What Does "Already Scaled" Mean?

`StandardScaler` transforms each feature so it has **mean = 0** and **standard deviation = 1** using:

$$z = \frac{x - \mu}{\sigma}$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $z$ | z-score | The scaled output value for a single data point |
| $x$ | raw value | The original, unscaled measurement (e.g., `alcohol = 11.5`) |
| $\mu$ | mu (mean) | The **average** of that feature across all training samples |
| $\sigma$ | sigma (std dev) | The **standard deviation** -- how spread out the values are around the mean |
| $x - \mu$ | deviation | How far this sample is from the average |
| $\frac{x - \mu}{\sigma}$ | normalized deviation | The deviation expressed in units of standard deviations |

**Step by step example** (feature: `alcohol`, mean=10.0, std=1.2):
$$z = \frac{11.5 - 10.0}{1.2} = \frac{1.5}{1.2} = 1.25$$

This means this wine's alcohol is **1.25 standard deviations above average**.

**Interpreting z-scores:**
- `0.0` = exactly average for that feature
- `+1.5` = 1.5 standard deviations **above** average
- `-2.0` = 2 standard deviations **below** average
- Most values fall roughly in the **-3 to +3 range** (~99.7% of data for a normal distribution)

> **Key distinction from MinMaxScaler:** MinMaxScaler forces values into [0, 1]. StandardScaler has **no hard cap** — values like `-1.455` or `1.798` are completely normal.

### Worked Example — Reading Real Values from `allwine-1.csv`

The data is already scaled, so each number in the file IS the z-score. Here is how to interpret real rows:

**Row 0** (`quality = 0`, bad wine):

| Feature | Scaled value | Interpretation |
|---------|-------------|----------------|
| fixed acidity | -0.7438 | 0.74 std devs **below** average acidity |
| volatile acidity | +0.8053 | 0.81 std devs **above** average |
| citric acid | -1.4559 | 1.46 std devs **below** average |
| residual sugar | -0.5415 | 0.54 std devs below average |
| alcohol | -1.2971 | 1.30 std devs **below** average alcohol |

**Row 3** (`quality = 1`, good wine):

| Feature | Scaled value | Interpretation |
|---------|-------------|----------------|
| fixed acidity | +1.3735 | 1.37 std devs **above** average acidity |
| volatile acidity | -1.5123 | 1.51 std devs **below** average (low = better) |
| citric acid | +1.4105 | 1.41 std devs above average |
| pH | -1.3973 | 1.40 std devs below average pH |
| alcohol | -0.9608 | 0.96 std devs below average alcohol |

**This is NOT the same as MinMaxScaler**, which caps values between 0 and 1. StandardScaler has no hard cap -- values like `-1.455` or `1.798` are completely normal.

**Why it matters here:** Since `allwine-1.csv` is already scaled, applying `StandardScaler` again in your code won't break anything (it'll just come out nearly identical), but it's technically redundant.

**Dataset shape:** 3198 rows x 11 columns (1488 bad, 1710 good)

> **DO NOT** use `sklearn.linear_model.LogisticRegression` or any other pre-built model. You are writing the math yourself.

---

## Background: What Is Logistic Regression?

Logistic regression is a way to predict a **yes/no (0 or 1) answer** from a set of numbers (your features).

It uses a special math function called the **sigmoid function** (also called the logistic function):

$$p(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_{10} x_{10})}}$$

Which is often written compactly as:

$$p(x) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \theta^T x = \sum_{j=0}^{10} \theta_j x_j$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $p(x)$ | predicted probability | Output: a number between 0 and 1. The model's confidence that this wine is good |
| $x$ | feature vector | The 10 wine measurements for one sample (e.g., acidity, alcohol, ...) |
| $x_j$ | j-th feature | One specific measurement (e.g., $x_1$ = fixed acidity, $x_2$ = volatile acidity, ...) |
| $\theta_j$ | theta j (weight) | The **learned weight** for feature $j$ — how much that feature matters |
| $\theta_0$ | bias (intercept) | A weight **not multiplied by any feature** — it shifts the decision boundary |
| $z$ | linear combination | The weighted sum of all features: $\theta_0 + \theta_1 x_1 + \cdots + \theta_{10} x_{10}$ |
| $e$ | Euler's number | Mathematical constant ≈ 2.718. Makes the function smooth and bounded |
| $e^{-z}$ | negative exponential | Flips $z$ so large $z$ → small value, small $z$ → large value |
| $\theta^T x$ | dot product | Shorthand for multiplying each weight by its feature and summing everything |

**Why the sigmoid shape?**

| $z$ value | $e^{-z}$ | $p(x)$ | Meaning |
|-----------|----------|--------|---------|
| very large (e.g. +10) | ≈ 0 | ≈ 1.0 | Very confident: good wine |
| 0 | 1.0 | 0.5 | Uncertain: coin flip |
| very negative (e.g. -10) | very large | ≈ 0.0 | Very confident: bad wine |

**In plain English:**
- Compute $z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_{10} x_{10}$ (weighted sum of all 10 features plus a bias)
- Plug $z$ into the sigmoid: $p = \frac{1}{1 + e^{-z}}$
- The output $p$ is your **predicted probability** that the wine is good (between 0 and 1)
- If $p > 0.5$, predict "good wine" (1). Otherwise predict "bad wine" (0)

### Worked Example — Sigmoid with Real Data from `allwine-1.csv`

Using **Row 3** (`quality = 1`, actual good wine). For demonstration, set all weights to $\theta_j = 0.1$ (not trained yet — just to show the calculation).

**Step 1 — Compute z (the linear combination):**

$$z = \theta_0 \cdot 1 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_{10} x_{10}$$

| Term | $\theta_j$ | $x_j$ | $\theta_j \cdot x_j$ |
|------|-----------|-------|---------------------|
| bias ($\theta_0$) | 0.1 | 1.0 | +0.1000 |
| fixed acidity | 0.1 | +1.3735 | +0.1374 |
| volatile acidity | 0.1 | -1.5123 | -0.1512 |
| citric acid | 0.1 | +1.4105 | +0.1410 |
| residual sugar | 0.1 | -0.5415 | -0.0542 |
| chlorides | 0.1 | -0.3556 | -0.0356 |
| free sulfur dioxide | 0.1 | +0.0292 | +0.0029 |
| density | 0.1 | -0.9741 | -0.0974 |
| pH | 0.1 | -1.3973 | -0.1397 |
| sulphates | 0.1 | -0.6403 | -0.0640 |
| alcohol | 0.1 | -0.9608 | -0.0961 |
| **Sum** | | | **z = -0.2569** |

**Step 2 — Plug z into the sigmoid:**

$$p = \frac{1}{1 + e^{-(-0.2569)}} = \frac{1}{1 + e^{0.2569}} = \frac{1}{1 + 1.2928} = \frac{1}{2.2928} = 0.4361$$

**Step 3 — Make prediction:**

$p = 0.4361 < 0.5$, so this model predicts **bad wine (0)**.
But the actual label is **y = 1 (good wine)** — the model got it wrong because $\theta = 0.1$ for all weights is not trained.
After training, the thetas will shift to values that push $p > 0.5$ for this sample.

**Your thetas ($\theta$) are what the model "learns".** Training the model means finding the best theta values that make the most correct predictions.

---

## Background: How Do You Train It? (Maximum Likelihood Estimation)

You adjust the theta values to maximize the **likelihood function**:

$$L(\theta) = \prod_{k: y_k=1} p_k \cdot \prod_{k: y_k=0} (1 - p_k)$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $L(\theta)$ | likelihood | A single number measuring "how good are my current theta values?" Higher is better |
| $\theta$ | theta (all weights) | The full set of 11 parameters ($\theta_0$ through $\theta_{10}$) being evaluated |
| $\prod$ | product (Π) | Multiply everything together (like $\Sigma$ means sum, $\Pi$ means multiply) |
| $k: y_k = 1$ | indices where label is 1 | Loop only over the wine samples that are **actually good** |
| $k: y_k = 0$ | indices where label is 0 | Loop only over the wine samples that are **actually bad** |
| $p_k$ | predicted probability | The sigmoid output for sample $k$ — how confident the model is that wine $k$ is good |
| $1 - p_k$ | complement probability | The model's confidence that wine $k$ is **bad** |

**Step by step:**
1. For each **actually good** wine ($y_k = 1$): take the model's predicted probability $p_k$. We want this to be close to 1.
2. For each **actually bad** wine ($y_k = 0$): take $(1 - p_k)$. We want this to be close to 1 too (meaning $p_k$ should be low).
3. Multiply all of those values together. If your thetas are good, every term is close to 1, so the product is close to 1.

**In practice, you maximize the log-likelihood** (easier math, same result — $\ln$ turns products into sums):

$$\ell(\theta) = \sum_{k=1}^{K} \left[ y_k \ln(p_k) + (1 - y_k) \ln(1 - p_k) \right]$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $\ell(\theta)$ | log-likelihood | The log of $L(\theta)$. Same goal (maximize), but sums instead of products |
| $\sum_{k=1}^{K}$ | summation | Add up the expression for every sample $k$ from 1 to $K$ (total number of samples) |
| $K$ | total samples | The number of wine samples in your dataset (e.g., 3198) |
| $y_k$ | true label | The actual answer for sample $k$: 1 = good wine, 0 = bad wine |
| $p_k$ | predicted probability | The sigmoid output $p(x_k)$ — the model's guess for sample $k$ |
| $\ln(p_k)$ | log of probability | Always negative (since $0 < p_k < 1$). Close to 0 when $p_k \approx 1$, very negative when $p_k \approx 0$ |
| $y_k \ln(p_k)$ | good-wine term | Only contributes when $y_k = 1$ (since $0 \times \text{anything} = 0$) |
| $(1 - y_k) \ln(1 - p_k)$ | bad-wine term | Only contributes when $y_k = 0$ |

**Why $y_k$ acts as a switch:**

| $y_k$ | Term active | Effect |
|-------|-------------|--------|
| $y_k = 1$ | $\ln(p_k)$ only | Rewards the model for predicting a HIGH probability for a wine that IS good |
| $y_k = 0$ | $\ln(1-p_k)$ only | Rewards the model for predicting a LOW probability for a wine that is NOT good |

### Worked Example — Log-Likelihood with Real Data from `allwine-1.csv`

Using **Row 0** (y=0, bad wine) and **Row 3** (y=1, good wine), with all $\theta_j = 0.1$:

From the sigmoid example above: $p_0 = 0.3847$, $p_3 = 0.4361$.

**Row 0** ($y_0 = 0$, bad wine):

$$y_0 \ln(p_0) + (1 - y_0)\ln(1 - p_0) = 0 \cdot \ln(0.3847) + 1 \cdot \ln(1 - 0.3847)$$
$$= 0 + \ln(0.6153) = -0.4857$$

*The $y_0 = 0$ kills the first term. We score how confidently the model said "bad" — $p_0 = 0.38$ is reasonably below 0.5, so the penalty is moderate.*

**Row 3** ($y_3 = 1$, good wine):

$$y_3 \ln(p_3) + (1 - y_3)\ln(1 - p_3) = 1 \cdot \ln(0.4361) + 0 \cdot \ln(0.5639)$$
$$= \ln(0.4361) + 0 = -0.8298$$

*The $(1-y_3) = 0$ kills the second term. The model predicted $p_3 = 0.44$ for a wine that IS good (should be > 0.5), so the penalty is high.*

**Total log-likelihood for these 2 samples:**

$$\ell = (-0.4857) + (-0.8298) = -1.3155$$

After training, $\ell$ will move toward 0 (less negative) as the model's predictions improve.

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

$$P(y=1 \mid x) = h_L(x) \cdot h_M(x) + h_R(x) \cdot (1 - h_M(x))$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $P(y=1 \mid x)$ | final probability | The ensemble's overall probability that wine $x$ is good |
| $x$ | feature vector | The 10 wine measurements for this sample |
| $h_M(x)$ | middle/router output | Sigmoid output of `LR_middle` — interpreted as "probability of routing LEFT" |
| $1 - h_M(x)$ | complement of router | "Probability of routing RIGHT" (the two must sum to 1) |
| $h_L(x)$ | left model output | Sigmoid output of `LR_left` — its probability that this wine is good |
| $h_R(x)$ | right model output | Sigmoid output of `LR_right` — its probability that this wine is good |
| $\cdot$ | multiply | Weights the classifier's output by how much probability flowed to it |

**Step by step for one wine sample:**

1. Feed the wine's features into `LR_middle` → get $h_M(x)$
2. $h_M$ gives the probability of routing LEFT; $(1 - h_M)$ is the probability of routing RIGHT
3. Feed the features into `LR_left` → get $h_L(x)$
4. Feed the features into `LR_right` → get $h_R(x)$
5. Combine into the weighted average formula
6. If result > 0.5, predict **good wine (1)**

**Why this formula makes sense:** It's a **probability-weighted average** of the two classifiers. If the router sends most samples left (high $h_M$), then `LR_left`'s answer dominates.

### Worked Example — Task 2 Ensemble with Real Data from `allwine-1.csv`

Using **Row 3** (`quality = 1`, actual good wine) with small random initialization thetas (seed=42, range ±0.1):

**Step 1 — Each model computes its own z and sigmoid output:**

| Model | $z = \theta^T x$ | $h = \sigma(z)$ | Role |
|-------|-----------------|-----------------|------|
| `LR_middle` | +0.0814 | **0.5203** | Router: 52% probability LEFT, 48% RIGHT |
| `LR_left` | +0.3575 | **0.5884** | Left classifier: 58.8% chance good wine |
| `LR_right` | +0.2316 | **0.5576** | Right classifier: 55.8% chance good wine |

**Step 2 — Combine into final probability:**

$$P(y=1) = h_L \cdot h_M + h_R \cdot (1 - h_M)$$

$$= 0.5884 \times 0.5203 + 0.5576 \times (1 - 0.5203)$$

$$= 0.5884 \times 0.5203 + 0.5576 \times 0.4797$$

$$= 0.3062 + 0.2675 = 0.5737$$

**Step 3 — Make prediction:**

$P(y=1) = 0.5737 > 0.5$ → predict **good wine (1)** ✓ (actual label: y=1, correct!)

**Step 4 — Verify P(y=0) = 1 - P(y=1):**

$$P(y=0) = (1-0.5884)\times 0.5203 + (1-0.5576)\times 0.4797 = 0.2142 + 0.2122 = 0.4264$$

$$P(y=1) + P(y=0) = 0.5737 + 0.4264 = 1.0000 \checkmark$$

And the probability of y=0 (bad wine):

$$P(y=0 \mid x) = (1 - h_L(x)) \cdot h_M(x) + (1 - h_R(x)) \cdot (1 - h_M(x))$$

**Verification that probabilities sum to 1:**

$$P(y=1) + P(y=0) = \bigl[h_L \cdot h_M + h_R \cdot (1-h_M)\bigr] + \bigl[(1-h_L) \cdot h_M + (1-h_R) \cdot (1-h_M)\bigr] = h_M + (1-h_M) = 1 \checkmark$$

## The Likelihood Function To Maximize:

$$L(\theta_M, \theta_L, \theta_R) = \prod_{i=1}^{n} P(y_i \mid x_i,\ \theta_M, \theta_L, \theta_R)$$

Which expands to the log-likelihood for optimization:

$$\ell = \sum_{i=1}^{n} \left[ y_i \ln P(y=1 \mid x_i) + (1 - y_i) \ln P(y=0 \mid x_i) \right]$$

**Symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $L(\theta_M, \theta_L, \theta_R)$ | joint likelihood | How well all three models together explain the training data |
| $\theta_M$ | middle weights | The 10 learned parameters for `LR_middle` (the router) |
| $\theta_L$ | left weights | The 10 learned parameters for `LR_left` |
| $\theta_R$ | right weights | The 10 learned parameters for `LR_right` |
| $\prod_{i=1}^{n}$ | product over $n$ samples | Multiply the probability term for every training sample |
| $n$ | number of samples | Total number of wine rows used for training |
| $P(y_i \mid x_i, \ldots)$ | per-sample probability | Uses the full combined formula: $h_L h_M + h_R(1-h_M)$ when $y_i=1$, or its complement when $y_i=0$ |
| $x_i$ | features of sample $i$ | The 10 wine measurements for training sample $i$ |
| $y_i$ | true label of sample $i$ | 1 = good wine, 0 = bad wine |

**Key point — 30 parameters, one joint optimization:**

| Model | Parameters | Description |
|-------|-----------|-------------|
| `LR_middle` ($\theta_M$) | 10 values | Controls how the data is split left/right |
| `LR_left` ($\theta_L$) | 10 values | Classifies the samples routed left |
| `LR_right` ($\theta_R$) | 10 values | Classifies the samples routed right |
| **Total** | **30 values** | All optimized simultaneously |

You concatenate all 30 thetas into one flat vector and pass it to `scipy.optimize.minimize` (or gradient ascent). The optimizer adjusts all 30 values together to maximize $\ell$.

### Worked Example — Task 2 Log-Likelihood with Real Data

Using the same Row 3 values from above ($P(y=1) = 0.5737$, $y_3 = 1$):

$$\ell_3 = y_3 \ln P(y=1 \mid x_3) + (1-y_3) \ln P(y=0 \mid x_3)$$

$$= 1 \cdot \ln(0.5737) + 0 \cdot \ln(0.4264) = \ln(0.5737) = -0.5548$$

For Row 0 ($y_0 = 0$, $P(y=1 \mid x_0)$ would be computed similarly with the 3 models):
Once computed, each sample contributes one term to the sum. Summing across all 3198 training rows gives the total $\ell$ that the optimizer maximizes.

**Interpretation:** $\ell_3 = -0.5548$ means the model is somewhat rewarded for Row 3 (it predicted 0.57 for a wine that IS good). A well-trained model might push this toward $\ln(0.95) \approx -0.05$ (near-zero penalty).

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

### Worked Example — Task 3 Ensemble with Real Data from `allwine-1.csv`

Using **Row 3** (`quality = 1`, actual good wine) with small random initialization thetas (seed=7, range ±0.1):

**Step 1 — All 7 nodes compute their sigmoid outputs:**

| Node | Layer | Role | $z = \theta^T x$ | $h = \sigma(z)$ |
|------|-------|------|-----------------|------------------|
| Node 1 | L1 | Router (top) | — | **h₁ = 0.5332** |
| Node 2 | L2 | Router (left branch) | — | **h₂ = 0.5062** |
| Node 3 | L2 | Router (right branch) | — | **h₃ = 0.4569** |
| Node 4 | L3 | Leaf — left of Node 2 | — | **h₄ = 0.5002** |
| Node 5 | L3 | Leaf — right of Node 2 | — | **h₅ = 0.5499** |
| Node 6 | L3 | Leaf — left of Node 3 | — | **h₆ = 0.4648** |
| Node 7 | L3 | Leaf — right of Node 3 | — | **h₇ = 0.5253** |

**Step 2 — Compute probability weights flowing to each leaf:**

| Leaf | Path from root | Weight formula | Value |
|------|---------------|----------------|-------|
| Node 4 | Root→Left→Left | $h_1 \times h_2$ | $0.5332 \times 0.5062 = 0.2699$ |
| Node 5 | Root→Left→Right | $h_1 \times (1 - h_2)$ | $0.5332 \times 0.4938 = 0.2633$ |
| Node 6 | Root→Right→Left | $(1-h_1) \times h_3$ | $0.4668 \times 0.4569 = 0.2133$ |
| Node 7 | Root→Right→Right | $(1-h_1) \times (1-h_3)$ | $0.4668 \times 0.5431 = 0.2535$ |
| **Total** | | | **1.0000** ✓ |

**Step 3 — Compute final probability:**

$$P(y=1) = h_4 \cdot w_4 + h_5 \cdot w_5 + h_6 \cdot w_6 + h_7 \cdot w_7$$

$$= 0.5002 \times 0.2699 + 0.5499 \times 0.2633 + 0.4648 \times 0.2133 + 0.5253 \times 0.2535$$

$$= 0.1350 + 0.1448 + 0.0991 + 0.1332 = 0.5121$$

**Step 4 — Make prediction:**

$P(y=1) = 0.5121 > 0.5$ → predict **good wine (1)** ✓ (actual label: y=1, correct!)

All 4 leaf probabilities are near 0.5 because the thetas are unoptimized random values. After training, the weights and leaf outputs will be tuned so that good wine samples confidently score > 0.5 and bad wine samples score < 0.5.

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
