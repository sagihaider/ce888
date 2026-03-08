# Lab 08: Dataset Shift & Transfer Learning

> **Course**: Data Science and Decision Making| **School**: Computer Science and Electronics Engineering  
> **Prepared by**: Dr Haider Raza, Senior Lecturer in AI, University of Essex  
> **Lab Duration**: 2 Hours |

---

## Learning Objectives

By the end of this lab you should be able to:

- [ ] Explain what dataset shift is and why it matters in real-world ML systems
- [ ] Identify and distinguish between Covariate Shift, Prior Probability Shift, and Concept Shift using their formal probability definitions
- [ ] Detect dataset shift programmatically using statistical techniques
- [ ] Apply importance weighting to correct for covariate shift
- [ ] Describe what transfer learning is and why it is used
- [ ] Choose an appropriate transfer learning strategy given dataset size and domain similarity
- [ ] Implement and evaluate a feature-extractor pipeline using a pre-trained ConvNet

---

## Prerequisites

Before starting this lab, make sure you are comfortable with:

- Basic Python (`numpy`, `matplotlib`)
- Probability fundamentals: joint probability P(X,Y), conditional probability P(Y|X)
- Supervised learning concepts: training set, test set, overfitting
- Neural network basics (what a convolutional layer does)

---

## Background Theory

### 1. Why Is Learning from Real-World Data Difficult?

Standard machine learning theory assumes that training and test data are **independent and identically distributed (i.i.d.)** — that is, both datasets are drawn from exactly the same underlying probability distribution P(X, Y).

In practice this assumption is routinely violated. Common culprits include:

| Problem | Description | Example |
|---------|-------------|---------|
| **Imbalanced data** | One class is vastly under-represented | Fraud detection (0.1% fraud cases) |
| **Overlapping classes** | Decision boundaries are ambiguous | Sentiment: "fine" can be positive or negative |
| **Label noise** | Some training labels are incorrect | Crowd-sourced annotation errors |
| **Dataset shift** | P_tr(X,Y) ≠ P_ts(X,Y) | Brain signal classification across subjects |

This lab focuses on the last problem: **dataset shift**.

---

### 2. Dataset Shift

> *Dataset shift appears when the joint distribution of inputs and outputs differs between the training stage and the test (deployment) stage.*

Formally, dataset shift is present when:

$$P_{\text{tr}}(X, Y) \neq P_{\text{ts}}(X, Y)$$

where:
- **X** — the feature vector (inputs / covariates)
- **Y** — the target variable (class label or output)
- **P_tr** — the distribution seen during training
- **P_ts** — the distribution seen at deployment / test time

**Why does this matter?**

A classifier trained to minimise loss on P_tr(X, Y) will often perform poorly when deployed under P_ts(X, Y) if the two distributions differ. The classifier has learned patterns that do not generalise to the new environment.

**Real-world examples:**

- A spam filter trained in 2020 faces new spam templates by 2024 → the language distribution shifts
- An ECG anomaly detector trained on one hospital's equipment is deployed in a different hospital with different sensors
- A financial fraud model trained on weekday transactions is evaluated on weekend transactions with very different spending patterns
- A face recognition system trained on studio-lit images is deployed in outdoor lighting

---

### 3. Types of Dataset Shift

The joint distribution P(X, Y) can be factored in two ways:

$$P(X, Y) = P(Y \mid X)\, P(X) \quad \text{(X → Y problems, predictive models)}$$

$$P(X, Y) = P(X \mid Y)\, P(Y) \quad \text{(Y → X problems, generative models)}$$

Each type of dataset shift is defined by which factor changes between training and test.

---

#### 3.1 Covariate Shift

**Applies to:** X → Y problems (predictive models)

**Definition:**

$$P_{\text{tr}}(Y \mid X) = P_{\text{ts}}(Y \mid X) \quad \text{and} \quad P_{\text{tr}}(X) \neq P_{\text{ts}}(X)$$

The relationship between inputs and outputs **stays the same**, but the marginal distribution of the inputs **changes**.

**Intuition:** Imagine training a regression model on houses in London, then deploying it in Manchester. The rule "more bedrooms → higher price" still holds (P(Y|X) is stable), but the distribution of house sizes is different (P(X) shifts).

**Correction:** Re-weight each training sample by the **importance ratio**:

$$w(x) = \frac{P_{\text{ts}}(x)}{P_{\text{tr}}(x)}$$

Samples that are rare in training but common at test time receive a higher weight, effectively re-balancing the training distribution toward the test distribution.

---

#### 3.2 Prior Probability Shift

**Applies to:** Y → X problems (generative models)

**Definition:**

$$P_{\text{tr}}(X \mid Y) = P_{\text{ts}}(X \mid Y) \quad \text{and} \quad P_{\text{tr}}(Y) \neq P_{\text{ts}}(Y)$$

The likelihood of observations given the class **stays the same**, but the **prior probability of the class** changes.

**Intuition:** A medical diagnostic model trained when disease prevalence is 10% is deployed in a region where prevalence is 40%. The symptoms given disease (P(X|Y)) are the same, but the base rate (P(Y)) has shifted. Naive Bayes classifiers are particularly sensitive to this.

---

#### 3.3 Concept Shift (Concept Drift)

**Definition (X → Y):**

$$P_{\text{tr}}(Y \mid X) \neq P_{\text{ts}}(Y \mid X) \quad \text{and} \quad P_{\text{tr}}(X) = P_{\text{ts}}(X)$$

The input distribution is **stable**, but the relationship between inputs and outputs **changes** — the very concept being learned has evolved.

**Intuition:** The word "tablet" referred to medication in 2005; by 2012 it primarily referred to a computing device. The feature distribution of text might be similar, but the correct label for the same word has changed.

This is the hardest type of shift to handle because no re-weighting strategy fixes it — the model must be retrained or continuously updated.

---

#### Summary Table

| Shift Type | What Changes | What Stays The Same | Problem Type |
|---|---|---|---|
| Covariate Shift | P(X) | P(Y\|X) | X → Y |
| Prior Probability Shift | P(Y) | P(X\|Y) | Y → X |
| Concept Shift | P(Y\|X) | P(X) | X → Y |
| — | P(X\|Y) | P(Y) | Y → X |

---

### 4. Causes of Dataset Shift

There are two primary root causes:

#### 4.1 Sample Selection Bias

The training data was collected through a **biased process** that does not represent the true deployment population.

- Example: A clinical trial recruits only urban patients; the model is deployed in rural settings
- Example: Web-scraped image datasets over-represent certain demographics

**Key point:** The bias is baked in at collection time and may not be apparent until deployment.

#### 4.2 Non-Stationary Environments

The world itself **changes over time or space**, making yesterday's training distribution an imperfect match for today's test distribution.

- Temporal change: user behaviour, language, market conditions
- Spatial change: sensor hardware differences across sites, regional demographic differences

**Correcting dataset shift generated by Sample Selection Bias:**

Standard cross-validation (CV) is almost unbiased under i.i.d. conditions, but it becomes **heavily biased under covariate shift** because the validation fold is drawn from the same biased distribution as the training fold — not from the true test distribution. Solutions include importance-weighted cross-validation.

---

### 5. Transfer Learning

> *"Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned."*  
> — Handbook of Research on Machine Learning Applications, 2009

> *"Transfer learning and domain adaptation refer to the situation where what has been learned in one setting is exploited to improve generalisation in another setting."*  
> — Goodfellow, Bengio, Courville — Deep Learning, 2016

#### Why do we need Transfer Learning?

Training deep convolutional networks from scratch requires:

- **Millions of labelled examples** (ImageNet has ~1.2 million images)
- **Weeks of compute time** on clusters of expensive GPUs
- **Significant engineering effort** to tune hyperparameters

In most real applications you will have only hundreds or thousands of labelled examples for your specific task. Transfer learning lets you leverage representations already learned on large datasets.

Andrew Ng (Stanford / Baidu) famously stated at NeurIPS 2016:

> *"After supervised learning, transfer learning will be the next driver of ML commercial success."*

---

### 6. Transfer Learning Strategies

There are three main strategies, and the right choice depends on:

1. **Dataset size** — how many labelled examples do you have for the target task?
2. **Domain similarity** — how similar is the source domain (e.g. ImageNet) to your target domain?

---

#### Strategy A: ConvNet as Fixed Feature Extractor

1. Take a ConvNet pre-trained on ImageNet (e.g. VGG-16)
2. **Remove the final fully-connected classification layer**
3. Feed your target-domain images through the frozen network
4. Collect the resulting feature vectors (e.g. 4096-D from VGG-16's penultimate layer)
5. Train a simple linear classifier (SVM or Softmax) on top of those features

**When to use:** Small dataset, similar domain  
**Why it works:** Early and middle layers of a ConvNet learn generic visual features (edges, textures, shapes) that transfer well to related visual tasks. Retraining them on a small dataset would cause overfitting.

---

#### Strategy B: Fine-tuning

1. Replace the classification head with a new one suited to your task
2. **Unfreeze some or all layers** of the pre-trained network
3. Train end-to-end with a small learning rate

**Fine-tune top layers only:** When dataset is medium-sized and similar to source domain  
**Fine-tune more layers:** When dataset is medium-sized but from a different domain  
**Fine-tune all layers:** When dataset is large and similar to source domain

**Why lower learning rates?** The pre-trained weights already encode useful representations. Large updates would destroy these features. A small learning rate gently nudges the weights toward the target distribution.

---

#### Strategy C: Train from Scratch

If your dataset is large **and** very different from ImageNet (e.g. satellite imagery, medical scans, EEG signals), it may be better to train a new network with random initialisation.

**When to use:** Large dataset, very different domain

---

## Key Formulae

| Concept | Formula |
|---|---|
| Dataset Shift | $P_{\text{tr}}(X,Y) \neq P_{\text{ts}}(X,Y)$ |
| Covariate Shift | $P_{\text{tr}}(Y\mid X) = P_{\text{ts}}(Y\mid X)$, $P_{\text{tr}}(X) \neq P_{\text{ts}}(X)$ |
| Prior Probability Shift | $P_{\text{tr}}(X\mid Y) = P_{\text{ts}}(X\mid Y)$, $P_{\text{tr}}(Y) \neq P_{\text{ts}}(Y)$ |
| Concept Shift | $P_{\text{tr}}(Y\mid X) \neq P_{\text{ts}}(Y\mid X)$, $P_{\text{tr}}(X) = P_{\text{ts}}(X)$ |
| Importance Weight | $w(x) = \dfrac{P_{\text{ts}}(x)}{P_{\text{tr}}(x)}$ |
| Weighted Accuracy | $\dfrac{\displaystyle\sum_{i=1}^{n} w_i \cdot \mathbf{1}[y_{\text{true},i} = \hat{y}_i]}{\displaystyle\sum_{i=1}^{n} w_i}$ |

---

## Quick Reference Decision Table

Use this table to decide which transfer learning strategy to apply:

```
                    ┌─────────────────────────────────────────┐
                    │       Domain Similarity to Source        │
                    ├──────────────────────┬──────────────────┤
                    │       Similar        │    Different      │
┌───────────────────┼──────────────────────┼──────────────────┤
│  Dataset  Small   │  Feature Extractor   │  Linear on top   │
│  Size     ───────-┼──────────────────────┼──────────────────┤
│           Medium  │  Fine-tune top layers│  Fine-tune more  │
│           ────────┼──────────────────────┼──────────────────┤
│           Large   │  Fine-tune all       │  Train from      │
│                   │  layers              │  Scratch         │
└───────────────────┴──────────────────────┴──────────────────┘
```

**Rule of thumb:** The more data you have and the more different your domain, the deeper you should fine-tune (or retrain from scratch).

---

## Further Reading

| Resource | Description |
|---|---|
| Storkey, A. (2009). *Dataset Shift in Machine Learning* | Foundational reference on all shift types |
| Sugiyama et al., JMLR (2007) | Covariate shift correction via importance weighting |
| Goodfellow, Bengio & Courville — *Deep Learning* (2016), Ch. 15 | Transfer learning and domain adaptation |
| Raza et al., *Soft Computing* (2015) | EEG covariate shift — real-world brain signal example |
| CS231n Stanford Notes — *Transfer Learning* | Excellent practical guide to fine-tuning ConvNets |
| Pan & Yang, *IEEE TKDE* (2010) | Survey of transfer learning — highly cited overview |

---
*University of Essex — School of Computer Science and Electronics Engineering*  
