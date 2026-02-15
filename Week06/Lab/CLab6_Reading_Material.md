# CE888 Lab 6 - Reading Material
## Time Series Analysis and Recommender Systems

**Prepared for:** MSc Data Science and Decision Making Students  
**Module:** CE888  
**Topic:** Introduction to Time Series and Collaborative Filtering  
**Reading Time:** 45-60 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Part A: Time Series Analysis](#part-a-time-series-analysis)
   - [What is Time Series Data?](#what-is-time-series-data)
   - [Trend Detection](#trend-detection)
   - [Smoothing Techniques](#smoothing-techniques)
   - [Forecasting Basics](#forecasting-basics)
3. [Part B: Recommender Systems](#part-b-recommender-systems)
   - [Introduction to Recommendations](#introduction-to-recommendations)
   - [User Similarity](#user-similarity)
   - [Collaborative Filtering](#collaborative-filtering)
   - [Evaluation Metrics](#evaluation-metrics)
4. [Worked Examples](#worked-examples)
5. [Common Patterns and Tips](#common-patterns-and-tips)
6. [Python Implementation Guide](#python-implementation-guide)
7. [Summary](#summary)

---

## Introduction

Welcome to Lab 6! In this lab, you'll learn two fundamental data science techniques:

1. **Time Series Analysis**: Understanding and predicting data that changes over time
2. **Recommender Systems**: Building systems that suggest items to users

These techniques power applications you use daily:
- **Time Series**: Stock market predictions, weather forecasts, sales forecasting
- **Recommender Systems**: Netflix recommendations, Amazon product suggestions, Spotify playlists

### Why These Topics Matter

Both techniques deal with **structured dependencies**:
- **Time series**: Past values influence future values
- **Recommender systems**: User preferences influence each other

Understanding these dependencies is key to making accurate predictions.

---

## Part A: Time Series Analysis

### What is Time Series Data?

**Definition:** Data points collected or recorded at successive time intervals.

**Examples:**
- Daily temperature readings
- Monthly sales figures
- Hourly website traffic
- Stock prices by minute

**Key Characteristic:** The **order matters**! Unlike regular datasets where you can shuffle rows, time series data must maintain its temporal sequence.

```python
# Regular data (order doesn't matter)
ages = [25, 30, 22, 28, 35]  # Can shuffle

# Time series data (order DOES matter)
temperatures = [20, 22, 21, 23, 24]  # Day 1, Day 2, Day 3, etc.
```

### Components of Time Series

Every time series has several components:

#### 1. **Trend**
The long-term increase or decrease in the data.

```
Example: Company revenue growing year over year
Year 1: $100K
Year 2: $120K
Year 3: $140K
Year 4: $160K  ← Clear upward trend
```

#### 2. **Seasonality**
Regular patterns that repeat at fixed intervals.

```
Example: Ice cream sales by month
Jan: Low
Feb: Low
...
Jun: High
Jul: High
Aug: High
Sep: Medium
...
Dec: Low
Pattern repeats every year!
```

#### 3. **Noise**
Random variation that can't be explained.

```
Example: Daily variations in temperature due to unpredictable factors
Expected: 20°C
Actual: 21°C (random +1°C variation)
```

### Trend Detection

**Why detect trends?**
- Understand if data is generally increasing, decreasing, or stable
- Decide what forecasting methods to use
- Identify business patterns

**Simple Method:**
Look at consecutive differences and see if most go in the same direction.

```python
data = [10, 12, 14, 16, 18, 20]
differences = [2, 2, 2, 2, 2]  # All positive → Upward trend!

data = [20, 19, 18, 17, 16, 15]
differences = [-1, -1, -1, -1, -1]  # All negative → Downward trend!

data = [10, 15, 12, 18, 14]
differences = [5, -3, 6, -4]  # Mixed → No clear trend
```

**Algorithm:**
1. Calculate all consecutive differences: `data[i+1] - data[i]`
2. Count positive vs negative differences
3. If > 70% go in same direction → trend exists

**Example Calculation:**

```python
data = [1, 3, 2, 4, 3, 5, 4, 6, 5, 7]
# Differences: [2, -1, 2, -1, 2, -1, 2, -1, 2]
# Positive: 6 (66.7%)
# Negative: 3 (33.3%)
# At 70% threshold: No trend
# At 60% threshold: Upward trend detected!
```

### Smoothing Techniques

**Purpose:** Remove noise to see underlying patterns more clearly.

#### Moving Average

**Concept:** Average values in a sliding window.

**Visual Example:**
```
Original data: [10, 20, 15, 25, 20, 30, 25]
Window size = 3

Position 0: [10, 20, 15] → Average = 15.0
Position 1: [20, 15, 25] → Average = 20.0
Position 2: [15, 25, 20] → Average = 20.0
Position 3: [25, 20, 30] → Average = 25.0
Position 4: [20, 30, 25] → Average = 25.0

Smoothed: [15.0, 20.0, 20.0, 25.0, 25.0]
```

**Effect:** The smoothed version shows the trend more clearly by reducing noise.

**Trade-off:**
- Larger window → smoother, but loses more detail
- Smaller window → keeps detail, but less smoothing

```python
Window = 2: More detail, less smooth
Window = 5: Very smooth, loses detail
Window = 10: Very smooth, loses a lot of detail
```

#### Exponential Smoothing

**Concept:** Give more weight to recent observations.

**Formula:**
```
smooth[0] = data[0]  (start with first value)
smooth[t] = α × data[t] + (1-α) × smooth[t-1]
```

**Parameter α (alpha):**
- α close to 1 (e.g., 0.9): React quickly to changes
- α close to 0 (e.g., 0.1): Smooth out changes

**Example with α = 0.5:**

```
data = [10, 20, 30, 40, 50]

smooth[0] = 10
smooth[1] = 0.5 × 20 + 0.5 × 10 = 10 + 5 = 15
smooth[2] = 0.5 × 30 + 0.5 × 15 = 15 + 7.5 = 22.5
smooth[3] = 0.5 × 40 + 0.5 × 22.5 = 20 + 11.25 = 31.25
smooth[4] = 0.5 × 50 + 0.5 × 31.25 = 25 + 15.625 = 40.625

Final smoothed value: 40.625
```

**Example with α = 0.9 (reacts quickly):**

```
data = [10, 10, 10, 100]  (sudden spike!)

smooth[0] = 10
smooth[1] = 0.9 × 10 + 0.1 × 10 = 10
smooth[2] = 0.9 × 10 + 0.1 × 10 = 10
smooth[3] = 0.9 × 100 + 0.1 × 10 = 90 + 1 = 91

Final: 91 (close to the spike!)
```

**Example with α = 0.1 (smooths changes):**

```
data = [10, 10, 10, 100]  (same spike!)

smooth[0] = 10
smooth[1] = 0.1 × 10 + 0.9 × 10 = 10
smooth[2] = 0.1 × 10 + 0.9 × 10 = 10
smooth[3] = 0.1 × 100 + 0.9 × 10 = 10 + 9 = 19

Final: 19 (smoothed out the spike!)
```

### Forecasting Basics

**Goal:** Predict future values based on past observations.

**Simple Approach - Moving Average Forecast:**
Use the average of recent values as the prediction.

```python
Historical data: [10, 20, 30, 40, 50]
Window = 3

Average of last 3 values: (30 + 40 + 50) / 3 = 40

Forecast:
- Next period: 40
- Period after that: 40
- Period after that: 40
(Same value for all future periods)
```

**Why use this?**
- Simple baseline
- Works well for stable series
- Easy to understand and explain

**When it fails:**
- Strong trends (keeps predicting old average)
- Seasonal patterns (doesn't capture cycles)

---

## Part B: Recommender Systems

### Introduction to Recommendations

**Goal:** Predict which items a user will like, even if they haven't seen them yet.

**The Challenge:**
```
        Movie1  Movie2  Movie3  Movie4  Movie5
Alice      5       ?       4       ?       2
Bob        4       5       ?       3       ?
Carol      ?       3       5       ?       4

? = Haven't watched yet
How do we fill in these blanks?
```

**Two Main Approaches:**

1. **Collaborative Filtering**: "Users who are similar to you liked these items"
2. **Content-Based**: "You liked these items, so you'll like similar items"

We'll focus on **Collaborative Filtering** in this lab.

### User Similarity

**Key Idea:** Find users with similar tastes.

**Why?**
If Alice and Bob rate movies similarly, we can:
- Recommend to Alice what Bob liked
- Recommend to Bob what Alice liked

#### Cosine Similarity

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)

Where:
A · B = dot product
||A|| = magnitude (length) of vector A
||B|| = magnitude (length) of vector B
```

**Intuitive Meaning:**
- Measures the "angle" between two rating vectors
- 1.0 = Perfect agreement (same angle)
- 0.0 = No similarity (perpendicular)
- Works well even when users rate on different scales

**Example 1: Perfect Similarity**

```python
Alice: [5, 4, 3, 2, 1]
Bob:   [5, 4, 3, 2, 1]

Dot product: 5×5 + 4×4 + 3×3 + 2×2 + 1×1 = 25 + 16 + 9 + 4 + 1 = 55
||Alice|| = √(25 + 16 + 9 + 4 + 1) = √55 ≈ 7.416
||Bob||   = √(25 + 16 + 9 + 4 + 1) = √55 ≈ 7.416

Similarity = 55 / (7.416 × 7.416) = 55 / 55 = 1.0

Perfect similarity! ✓
```

**Example 2: With Missing Values**

```python
Alice: [5, NaN, 3, NaN, 1]
Bob:   [5,  4,  3,  2, NaN]

Step 1: Find common rated items
Common positions: 0 and 2
Alice (common): [5, 3]
Bob (common):   [5, 3]

Step 2: Calculate on common items only
Dot product: 5×5 + 3×3 = 25 + 9 = 34
||Alice|| = √(25 + 9) = √34 ≈ 5.831
||Bob||   = √(25 + 9) = √34 ≈ 5.831

Similarity = 34 / (5.831 × 5.831) = 34 / 34 = 1.0

Still perfect! ✓
```

**Example 3: Partial Similarity**

```python
Alice: [5, 4, 3]
Bob:   [5, 3, 2]

Dot product: 5×5 + 4×3 + 3×2 = 25 + 12 + 6 = 43
||Alice|| = √(25 + 16 + 9) = √50 ≈ 7.071
||Bob||   = √(25 + 9 + 4) = √38 ≈ 6.164

Similarity = 43 / (7.071 × 6.164) = 43 / 43.58 ≈ 0.987

Very similar (98.7% similar) ✓
```

**Example 4: No Common Ratings**

```python
Alice: [5, NaN, NaN]
Bob:   [NaN, 4, 3]

No common positions!
Similarity = 0.0 (by convention)
```

### Collaborative Filtering

**Concept:** Use ratings from similar users to predict missing ratings.

#### The Process

```
Step 1: Find similar users
Step 2: Get their ratings for the target item
Step 3: Combine using weighted average
```

**Detailed Example:**

```
Goal: Predict Alice's rating for Movie 3

        Movie1  Movie2  Movie3
Alice      5       3       ?      ← Need to predict this!
Bob        5       3       4      (similarity = 1.0)
Carol      4       2       5      (similarity = 0.9)
Dave       3       1       3      (similarity = 0.7)
Eve        2       5       2      (similarity = 0.3)

Step 1: Find similar users
We'll use top k=2 users: Bob and Carol

Step 2: Get their ratings for Movie 3
Bob rated it: 4
Carol rated it: 5

Step 3: Weighted average
Prediction = (1.0 × 4 + 0.9 × 5) / (1.0 + 0.9)
           = (4.0 + 4.5) / 1.9
           = 8.5 / 1.9
           = 4.47

Alice will probably rate Movie 3 as 4.47 ≈ 4.5 stars
```

**Why Weighted Average?**
- Trust similar users more (higher weight)
- Less similar users have less influence (lower weight)
- More fair than simple average

**Example with Simple Average (Wrong!):**
```
Simple average = (4 + 5) / 2 = 4.5

But this ignores that Bob is MORE similar than Carol!
```

**Example with Weighted Average (Correct!):**
```
Weighted = (1.0×4 + 0.9×5) / (1.0 + 0.9) = 4.47

Bob's opinion matters slightly more because he's more similar.
```

#### K-Nearest Neighbors (KNN)

**Parameter k:** How many similar users to consider

**Example: Different k values**

```
All neighbors with ratings:
User A: similarity=0.95, rating=5
User B: similarity=0.90, rating=4
User C: similarity=0.85, rating=5
User D: similarity=0.80, rating=3
User E: similarity=0.75, rating=4

k=1: Use only User A
Prediction = 5.0

k=2: Use Users A and B
Prediction = (0.95×5 + 0.90×4) / (0.95+0.90) = 4.51

k=3: Use Users A, B, and C
Prediction = (0.95×5 + 0.90×4 + 0.85×5) / (0.95+0.90+0.85) = 4.67

k=5: Use all 5 users
Prediction = (0.95×5 + 0.90×4 + 0.85×5 + 0.80×3 + 0.75×4) / (sum of sims) = 4.30
```

**Choosing k:**
- Small k (1-3): More personalized, but potentially noisy
- Large k (10+): More stable, but less personalized
- Typical choice: k=5 to k=20

### Evaluation Metrics

**How do we know if our recommender system is good?**

#### Root Mean Squared Error (RMSE)

**Purpose:** Measure how far predictions are from actual ratings.

**Formula:**
```
RMSE = √(mean((actual - predicted)²))
```

**Step-by-step Example:**

```python
Actual:    [5,  4,  3,  2,  1]
Predicted: [4,  4,  3,  3,  1]

Step 1: Calculate errors
Errors: [5-4, 4-4, 3-3, 2-3, 1-1] = [1, 0, 0, -1, 0]

Step 2: Square the errors
Squared: [1², 0², 0², (-1)², 0²] = [1, 0, 0, 1, 0]

Step 3: Calculate mean
Mean = (1 + 0 + 0 + 1 + 0) / 5 = 2/5 = 0.4

Step 4: Take square root
RMSE = √0.4 = 0.632
```

**Interpretation:**
- RMSE = 0: Perfect predictions
- RMSE = 0.5: Pretty good (on 1-5 scale)
- RMSE = 1.0: Okay
- RMSE = 2.0: Poor

**Why Square Errors?**
- Penalizes large errors more than small errors
- Error of 2 is more than twice as bad as error of 1

```
Error = 1: Squared = 1
Error = 2: Squared = 4 (4× worse!)
Error = 3: Squared = 9 (9× worse!)
```

**With Missing Values:**

```python
Actual:    [5,  NaN,  3,  NaN,  1]
Predicted: [4,   2,   3,   4,   1]

Step 1: Only use non-NaN positions
Valid positions: 0, 2, 4
Actual (valid): [5, 3, 1]
Predicted (valid): [4, 3, 1]

Step 2: Calculate RMSE on valid data
Errors: [1, 0, 0]
Squared: [1, 0, 0]
Mean: 1/3 ≈ 0.333
RMSE: √0.333 ≈ 0.577
```

---

## Worked Examples

### Example 1: Complete Time Series Analysis

**Scenario:** Daily ice cream sales for one week.

```python
data = [100, 120, 110, 130, 140, 160, 150]  # Units sold

# Task 1: Detect trend
differences = [20, -10, 20, 10, 20, -10]
positive = 4 (66.7%)
negative = 2 (33.3%)

At 60% threshold: Trend detected! (Upward)
At 70% threshold: No trend (not enough positive)

# Task 2: Moving average (window=3)
[100, 120, 110] → 110.0
[120, 110, 130] → 120.0
[110, 130, 140] → 126.67
[130, 140, 160] → 143.33
[140, 160, 150] → 150.0

Smoothed: [110.0, 120.0, 126.67, 143.33, 150.0]

# Task 3: Exponential smoothing forecast (α=0.5, horizon=2)
smooth[0] = 100
smooth[1] = 0.5×120 + 0.5×100 = 110
smooth[2] = 0.5×110 + 0.5×110 = 110
smooth[3] = 0.5×130 + 0.5×110 = 120
smooth[4] = 0.5×140 + 0.5×120 = 130
smooth[5] = 0.5×160 + 0.5×130 = 145
smooth[6] = 0.5×150 + 0.5×145 = 147.5

Forecast for next 2 days: [147.5, 147.5]
```

### Example 2: Complete Recommender System

**Scenario:** Movie rating prediction

```python
# Rating matrix (1-5 stars)
        Movie1  Movie2  Movie3  Movie4
Alice      5       4       ?       2
Bob        5       5       4       1
Carol      4       3       5       2
Dave       5       4       3       2

# Task 1: Find Alice's similarity with others

Alice vs Bob:
Common movies: 1, 2, 4
Alice: [5, 4, 2]
Bob:   [5, 5, 1]
Dot: 5×5 + 4×5 + 2×1 = 25 + 20 + 2 = 47
||Alice|| = √(25+16+4) = √45 ≈ 6.708
||Bob|| = √(25+25+1) = √51 ≈ 7.141
Similarity = 47 / (6.708 × 7.141) = 0.981

Alice vs Carol:
Common movies: 1, 2, 4
Alice: [5, 4, 2]
Carol: [4, 3, 2]
Dot: 5×4 + 4×3 + 2×2 = 20 + 12 + 4 = 36
||Alice|| = √45 ≈ 6.708
||Carol|| = √(16+9+4) = √29 ≈ 5.385
Similarity = 36 / (6.708 × 5.385) = 0.997

Alice vs Dave:
Common movies: 1, 2, 4
Alice: [5, 4, 2]
Dave:  [5, 4, 2]
Perfect match!
Similarity = 1.000

# Task 2: Predict Alice's rating for Movie 3

All users rated Movie 3:
Bob: 4 (similarity=0.981)
Carol: 5 (similarity=0.997)
Dave: 3 (similarity=1.000)

Using k=2 (top 2 neighbors):
Dave and Carol

Weighted prediction:
= (1.000×3 + 0.997×5) / (1.000 + 0.997)
= (3.000 + 4.985) / 1.997
= 7.985 / 1.997
= 4.00

Alice will probably rate Movie 3 as 4.0 stars

# Task 3: Calculate RMSE

Suppose our predictions were:
Actual:    [4, 5, 3]  (for Bob, Carol, Dave on Movie 3)
Predicted: [4, 4.5, 3]

Errors: [0, 0.5, 0]
Squared: [0, 0.25, 0]
Mean: 0.25 / 3 ≈ 0.083
RMSE: √0.083 ≈ 0.289

Very good! (RMSE < 0.5)
```

---

## Common Patterns and Tips

### Time Series Patterns

**Pattern 1: Detect upward/downward trend**
```python
import numpy as np

def has_trend(data, threshold=0.7):
    diffs = np.diff(data)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    total = len(diffs)
    return (pos/total >= threshold) or (neg/total >= threshold)
```

**Pattern 2: Sliding window**
```python
def sliding_window_operation(data, window, operation):
    results = []
    for i in range(len(data) - window + 1):
        window_data = data[i:i+window]
        result = operation(window_data)
        results.append(result)
    return results
```

**Pattern 3: Iterative smoothing**
```python
def iterative_smooth(data, alpha):
    smooth = data[0]
    for value in data[1:]:
        smooth = alpha * value + (1 - alpha) * smooth
    return smooth
```

### Recommender System Patterns

**Pattern 1: Filter NaN values**
```python
import numpy as np

def filter_common(arr1, arr2):
    mask = ~(np.isnan(arr1) | np.isnan(arr2))
    return arr1[mask], arr2[mask]
```

**Pattern 2: Weighted average**
```python
def weighted_average(values, weights):
    numerator = sum(w * v for w, v in zip(weights, values))
    denominator = sum(weights)
    return numerator / denominator if denominator > 0 else None
```

**Pattern 3: Top-k selection**
```python
def get_top_k(items, scores, k):
    # Create tuples of (score, item)
    paired = list(zip(scores, items))
    # Sort by score descending
    paired.sort(key=lambda x: x[0], reverse=True)
    # Take top k
    return paired[:k]
```

---

## Python Implementation Guide

### Essential Libraries

```python
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation (optional)
```

### Numpy Quick Reference

**Arrays:**
```python
# Create array
arr = np.array([1, 2, 3, 4, 5])

# Array operations
arr + 5       # Add 5 to all elements
arr * 2       # Multiply all by 2
arr ** 2      # Square all elements

# Statistics
np.mean(arr)  # Average
np.sum(arr)   # Sum
np.std(arr)   # Standard deviation
```

**Differences:**
```python
data = np.array([10, 20, 15, 25])
diffs = np.diff(data)  # [10, -5, 10]
```

**Boolean operations:**
```python
arr = np.array([1, 2, 3, 4, 5])
arr > 3              # [False, False, False, True, True]
np.sum(arr > 3)      # 2 (count of True)
```

**NaN handling:**
```python
arr = np.array([1, np.nan, 3, np.nan, 5])

np.isnan(arr)        # [False, True, False, True, False]
~np.isnan(arr)       # [True, False, True, False, True] (NOT)
np.nanmean(arr)      # 3.0 (ignores NaN)
```

**Masking:**
```python
arr = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
arr[mask]            # [1, 3, 5] (only True positions)
```

**Vector operations:**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)         # Dot product: 1×4 + 2×5 + 3×6 = 32
np.linalg.norm(a)    # Magnitude: √(1² + 2² + 3²) = √14
```

### Common Operations

**Rounding:**
```python
value = 3.14159
round(value, 2)      # 3.14 (2 decimal places)
round(value, 3)      # 3.142 (3 decimal places)
```

**List comprehensions:**
```python
# Create list of squared values
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]  # [1, 4, 9, 16, 25]

# With condition
positive = [x for x in data if x > 2]  # [3, 4, 5]

# With transformation and rounding
rounded = [round(x * 0.5, 2) for x in data]  # [0.5, 1.0, 1.5, 2.0, 2.5]
```

**Sorting:**
```python
# Sort list of tuples by first element
pairs = [(0.9, 5), (0.7, 3), (1.0, 4)]
pairs.sort(key=lambda x: x[0], reverse=True)
# Result: [(1.0, 4), (0.9, 5), (0.7, 3)]
```

---

## Summary

### Time Series Key Points

1. **Order matters** - Don't shuffle time series data
2. **Trends** - Check if data generally increases/decreases
3. **Smoothing** - Remove noise to see patterns
   - Moving average: Simple average of window
   - Exponential smoothing: Weight recent data more
4. **Forecasting** - Use past to predict future
   - Simple: Use average of recent values

### Recommender Systems Key Points

1. **Collaborative filtering** - Similar users like similar items
2. **Similarity** - Measure how alike two users are
   - Cosine similarity: 0 to 1 scale
   - Ignore missing ratings (NaN)
3. **KNN prediction** - Use k most similar users
   - Weighted average by similarity
   - Higher similarity = more influence
4. **RMSE** - Measure prediction accuracy
   - Lower is better
   - Square root of mean squared errors

### Success Checklist

Before starting the graded lab:

- [ ] I understand what time series data is
- [ ] I can calculate consecutive differences
- [ ] I can compute moving averages
- [ ] I understand exponential smoothing
- [ ] I know how to calculate cosine similarity
- [ ] I can handle NaN values in ratings
- [ ] I understand weighted averages
- [ ] I can calculate RMSE
- [ ] I've practiced with the examples
- [ ] I've attempted the practice questions

---

## Next Steps

1. **Read through all examples** carefully
2. **Work through the practice questions** (next document)
3. **Check your solutions** against provided answers
4. **Ask questions** if anything is unclear
5. **Attempt the graded lab** when you feel confident

**Good luck! You've got this! 🚀**

---

*End of Reading Material*

**Prepared by:** CE888 Teaching Team  
**Last Updated:** February 2026  
**Questions?** Post in the discussion forum or attend office hours
