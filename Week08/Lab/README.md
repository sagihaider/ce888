# 🧠 CE888 – Lab 5: Learning from Sequences with Recurrent Neural Networks

> **Module:** CE888 Data Science and Decision Making  
> **Topic:** Recurrent Neural Networks (RNNs), LSTMs, and Text Processing  
> **Lab Duration:** 2 Hours 

---

## 📋 Before You Start

Make sure you are comfortable with:
- NumPy array operations and matrix multiplication (`@`)
- Basic Python classes (`__init__`, methods)
- The concept of a neural network layer and what "training" means

If you need a refresher on any of these, spend 10 minutes on it before opening the notebook.

---

## Topics to Review Before / During the Lab

### 1. What is a Sequence?

A **sequence** is any ordered collection of data where position matters:

- A time series of daily temperatures: `[12.1, 13.4, 11.8, ...]`
- A sentence represented as word indices: `[4, 27, 3, 91, ...]`
- A patient's heart rate measured every second

The key property is that **earlier elements influence how we interpret later ones**. A regular Dense layer sees each input independently — it has no memory. An RNN is designed to process elements **one at a time**, carrying a hidden state (memory) forward through the sequence.

---

### 2. The SimpleRNN Cell — Core Equation

At every timestep `t`, a SimpleRNN cell computes:

```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
```

| Symbol | Shape | Role |
|--------|-------|------|
| `x_t` | `(input_features,)` | Input at this timestep |
| `h_{t-1}` | `(hidden_units,)` | Hidden state from previous step |
| `W_x` | `(hidden_units, input_features)` | Input weight matrix |
| `W_h` | `(hidden_units, hidden_units)` | Recurrent weight matrix |
| `b` | `(hidden_units,)` | Bias vector |
| `h_t` | `(hidden_units,)` | New hidden state (also the output) |

> **Key insight:** `h_0` (the very first hidden state) is initialised to a vector of **zeros**.  
> **Key insight:** The same weights `W_x`, `W_h`, `b` are **shared** across every timestep — that is why RNNs can handle variable-length sequences.

---

### 3. Trainable Parameter Counts

Understanding how many parameters a layer has tells you how complex it is and how much data you need to train it well.

#### SimpleRNN
```
params = (input_features × H) + (H × H) + H
       =  W_x                +  W_h     + b
```

#### LSTM (Long Short-Term Memory)
An LSTM has **four gates** (input, forget, cell, output). Each gate has the same structure as a SimpleRNN cell:
```
params = 4 × [ (input_features × H) + (H × H) + H ]
       = 4 × SimpleRNN_params
```

> **Why 4×?** The four gates give the LSTM its ability to selectively remember and forget, solving the **vanishing gradient problem** that makes SimpleRNN forget long-range dependencies.

#### Stacked RNNs
When you stack RNN layers, the **input size of each layer after the first equals the hidden units of the previous layer**:

```
Layer 1: input = input_features,   output = H1
Layer 2: input = H1,               output = H2
Layer 3: input = H2,               output = H3
```

---

### 4. Data Shape Through an RNN

RNN layers in Keras expect inputs of shape `(batch_size, timesteps, features)`.

The output shape depends on `return_sequences`:

```
return_sequences=False  →  (batch_size, hidden_units)          # only last state
return_sequences=True   →  (batch_size, timesteps, hidden_units) # all states
```

> **Rule:** When **stacking** RNN layers, every layer **except the last** must use `return_sequences=True` — otherwise the next layer has no sequence to process.

---

### 5. Padding Sequences

Real text data has variable-length sequences. Keras needs fixed-size batches, so we **pad** short sequences with zeros and **truncate** long ones.

```python
# Short sequence: [1, 2]        → [1, 2, 0, 0, 0]  (post-padding, maxlen=5)
# Long sequence:  [1,2,3,4,5,6] → [1, 2, 3, 4, 5]  (post-truncation, maxlen=5)
```

The Keras utility `pad_sequences(sequences, maxlen, padding='post', truncating='post')` handles this in one call.

---

### 6. Text Preprocessing Pipeline

Converting raw text to something an RNN can process involves three steps:

```
Raw text  →  Tokenisation  →  Indexing  →  Padding  →  Embedding  →  RNN
"I love ML"  ["i","love","ml"]  [3, 7, 12]  [3,7,12,0,0]  vectors    h_t
```

**Tokenisation** — split text into units (words, characters, n-grams)  
**Indexing** — assign a unique integer to each token (`word_to_index` dictionary)  
**Padding** — make all sequences the same length  
**Embedding** — map each integer to a dense vector (learned during training)

> **Unknown words:** Any word seen at inference time but not during training maps to index `0` (`<UNK>`).

---

### 7. Keras Model Architectures

#### Stacked SimpleRNN (Time-series regression)
```python
model = keras.Sequential([
    layers.SimpleRNN(32, return_sequences=True, input_shape=(timesteps, features)),
    layers.SimpleRNN(16, return_sequences=True),
    layers.SimpleRNN(8),   # last layer: return_sequences=False
    layers.Dense(1)        # regression output
])
model.compile(optimizer='adam', loss='mse')
```

#### LSTM Text Classifier
```python
model = keras.Sequential([
    layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    layers.LSTM(128),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary classification
])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

> **Dropout in RNNs:** Do NOT use a standalone `Dropout` layer between recurrent layers — it hurts performance. Use the built-in `dropout` and `recurrent_dropout` arguments instead. A `Dropout` layer after the final RNN (before Dense) is fine.

---

### 8. SimpleRNN vs LSTM vs GRU — When to Use What

| Layer | Pros | Cons | Use when |
|-------|------|------|----------|
| `SimpleRNN` | Easy to understand, fast | Forgets long sequences (vanishing gradient) | Short sequences, learning/teaching |
| `LSTM` | Handles long-range dependencies | 4× more parameters, slower | Long text, complex time series |
| `GRU` | Faster than LSTM, fewer params | Slightly less expressive | Limited data, speed matters |

---

## 🔑 Key Concepts Cheat Sheet

```
@           →  matrix multiplication (same as np.dot for 2D)
tanh(x)     →  activation: squashes values to (-1, 1)
h_t         →  hidden state at timestep t (the RNN's "memory")
W_x         →  weights applied to the input
W_h         →  weights applied to the previous hidden state (recurrence!)
return_sequences=True   →  pass ALL hidden states to the next layer
return_sequences=False  →  pass ONLY the last hidden state forward
pad_sequences           →  make all sequences the same length with zeros
Embedding   →  learnable lookup table: integer index → dense vector
```

---

## 📝 Graded Questions Summary

| # | Topic | Difficulty | Points | Keras? |
|---|-------|-----------|--------|--------|
| Q1 | Simulate RNN Forward Pass | Easy | 10 | ❌ |
| Q2 | Pad Variable-Length Sequences | Easy | 10 | ✅ |
| Q3 | Build Stacked SimpleRNN Model | Medium | 15 | ✅ |
| Q4 | Build LSTM Text Classifier | Medium | 15 | ✅ |
| Q5 | Compute RNN Output Shape | Easy | 10 | ❌ |
| Q6 | Count SimpleRNN Stack Parameters | Medium | 15 | ❌ |
| Q7 | Word-Level Tokenizer from Scratch | Medium | 15 | ❌ |
| Q8 | Compare LSTM vs SimpleRNN Params | Medium | 15 | ❌ |
| | **Total** | | **105** | |

> ✅ = requires TensorFlow/Keras  
> ❌ = pure Python / NumPy only

---

## 💡 Tips for the Graded Questions

1. **Read the docstring carefully** — the expected input/output types (list vs array, int vs float) are specified there.
2. **Q1:** Don't forget `h_0 = zeros`. The initial state must exist before the loop starts.
3. **Q3:** The most common mistake is forgetting `return_sequences=True` on intermediate layers. If you only have one RNN layer, `return_sequences` doesn't matter.
4. **Q6 & Q8:** Work out the maths by hand first on paper, then code it — it's easier to verify.
5. **Q7:** Remember that `0` is reserved for unknown words. Real word indices start at `1`.
6. **Q8:** The ratio between LSTM and SimpleRNN parameters is always **exactly 4.0** regardless of layer size — make sure your formula reflects that.

---

## 📖 Further Reading (Optional)

- [Keras RNN Guide](https://keras.io/guides/working_with_rnns/)
- [Understanding LSTM Networks – Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning with Python, 2nd Ed – François Chollet (Chapter 10)](https://www.manning.com/books/deep-learning-with-python-second-edition)
- [Keras TextVectorization](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)

---

*CE888 – University of Essex, CSEE | Dr. Ana Matran-Fernandez*
