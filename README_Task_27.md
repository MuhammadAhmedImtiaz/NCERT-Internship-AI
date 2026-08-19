# Task 27 – NLP Basics: Tokenization, Word Embeddings & Hugging Face Transformers

## Part A: Tokenization Foundations & Byte-Pair Encoding (BPE)

### 1. Tokenization Paradigms Comparison
| Paradigm | Typical Vocabulary Size | OOV Risk | Average Sequence Length |
| :--- | :--- | :--- | :--- |
| **Word-Level** | Very Large ($>100\text{k}$) | High (frequent `<UNK>`) | Short |
| **Character-Level** | Very Small ($50 - 256$) | Zero (No OOV) | Very Long (high compute) |
| **Subword-Level** | Optimal ($30\text{k} - 50\text{k}$) | Zero (rare words broken into subwords) | Balanced |

### 2. Byte-Pair Encoding (BPE) Algorithmic Formulation
BPE is a data-driven compression algorithm adapted for subword segmentation:
1. Initialize vocabulary $V$ with all unique characters in the training corpus, appending an end-of-word marker `</w>`.
2. Iteratively count the frequency of all adjacent symbol pairs $(c_i, c_j)$ across the tokenized corpus.
3. Identify the most frequent pair:
   $$(c_A, c_B) = \arg\max_{c_i, c_j} \text{Count}(c_i, c_j)$$
4. Merge $(c_A, c_B)$ into a single new subword token $c_A c_B$, add it to $V$, and update the corpus.
5. Repeat for $K$ merge iterations until the target vocabulary size is reached.

---

## Part B: Word Embeddings (Word2Vec & GloVe Formulations)

### 1. Word2Vec Skip-Gram & Negative Sampling Derivation
Given a center word $w_t$ and context window $C$, the Skip-gram objective maximizes the log-likelihood of context words $w_{t+j}$:
$$\mathcal{L}_{\text{Skip-gram}} = \sum_{t=1}^T \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$$

#### Full Softmax Formulation:
$$P(w_O \mid w_I) = \frac{\exp(v'_{w_O}{}^T v_{w_I})}{\sum_{w=1}^{|V|} \exp(v'_w{}^T v_{w_I})}$$
*Bottleneck:* Calculating the denominator requires summing over all $|V|$ words in the vocabulary ($O(|V|)$ complexity), which is computationally prohibitive for large vocabularies.

#### Negative Sampling Approximation ($\text{SGNS}$):
To bypass the full partition function, Negative Sampling reformulates the task as a binary logistic classification problem (distinguishing true context words from $k$ noise samples drawn from distribution $P_n(w) \propto f(w)^{3/4}$):
$$\mathcal{L}_{\text{SGNS}} = \log \sigma(v'_{w_O}{}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v'_{w_i}{}^T v_{w_I}) \right]$$
*Complexity:* Reduces computation from $O(|V|)$ to $O(k)$, where $k \ll |V|$ (typically $k \in [5, 20]$).

### 2. GloVe (Global Vectors) Formulation
GloVe trains on the global co-occurrence matrix $X$, where $X_{ij}$ denotes the number of times word $j$ appears in the context of word $i$. It minimizes the weighted least-squares loss:
$$J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
Where the weighting function $f(X_{ij}) = \min\left(1, (X_{ij} / x_{\max})^\alpha\right)$ prevents frequent stop words from dominating the loss.

---

## Part C & D: Quantitative Findings & Polysemy Dynamics

### 1. Static vs. Contextual Embeddings Benchmark
| Representation Type | Polysemy Handling | Cosine Similarity Across Contexts | OOV Resolution |
| :--- | :--- | :--- | :--- |
| **Static (Word2Vec / GloVe)** | Fixed 1-to-1 vector | $1.0000$ (No variation) | Replaced with `<UNK>` |
| **Contextual (DistilBERT)** | Dynamic contextual state | $\approx 0.68 - 0.76$ | Subword decomposition |

### 2. Polysemous Shift Observation
When comparing the token **"bank"** between financial (*"He deposited his money at the bank branch."*) and natural river (*"They walked along the river bank during sunset."*) contexts:
- Static embeddings assign identical representations ($S_{\cos} = 1.0$).
- DistilBERT's contextual attention alters the final hidden state dynamically, showing a lower cosine similarity ($S_{\cos} \approx 0.72$), demonstrating distinct semantic subspace activation.

### 3. Pipeline Connection to Task 26 Transformer Architecture
Subword Tokenization $\to$ Input Embedding Matrix Lookup $\to$ Positional Encoding Addition $\to$ Input to Multi-Head Self-Attention Stacks.