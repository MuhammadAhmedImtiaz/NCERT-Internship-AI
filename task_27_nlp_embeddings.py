# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 27: TOKENIZATION, WORD EMBEDDINGS (WORD2VEC) & HUGGING FACE
# =====================================================================

import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
sns.set_theme(style="whitegrid")

# =====================================================================
# PART A: SIMPLIFIED FROM-SCRATCH BYTE-PAIR ENCODING (BPE)
# =====================================================================
print("""
===================================================================
PART A: FROM-SCRATCH BYTE-PAIR ENCODING (BPE) TOKENIZER
===================================================================
""")

corpus = [
    "low lower lowest",
    "newer newest",
    "wide wider widest",
    "smart smarter smartest"
]

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

word_counts = collections.defaultdict(int)
for line in corpus:
    for word in line.split():
        word_formatted = " ".join(list(word)) + " </w>"
        word_counts[word_formatted] += 1

bpe_vocab = dict(word_counts)
num_merges = 10
merge_rules = []

print("Initial Base Vocabulary:")
print(bpe_vocab)

for i in range(num_merges):
    pairs = get_stats(bpe_vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    bpe_vocab = merge_vocab(best_pair, bpe_vocab)
    merge_rules.append(best_pair)
    print(f"Merge Step {i+1:02d}: Merged pair {best_pair} -> '{best_pair[0] + best_pair[1]}'")

print(f"\nFinal BPE Vocabulary (after {num_merges} merges):")
print(bpe_vocab)

def encode_bpe(word, rules):
    word_tokens = list(word) + ["</w>"]
    for pair in rules:
        i = 0
        new_tokens = []
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        word_tokens = new_tokens
    return word_tokens

test_word = "lowest"
encoded_subwords = encode_bpe(test_word, merge_rules)
print(f"\nEncoding sample word '{test_word}': {encoded_subwords}")

# =====================================================================
# PART B: PYTORCH WORD2VEC (SKIP-GRAM) & PCA VISUALIZATION
# =====================================================================
print("""
===================================================================
PART B: WORD2VEC (SKIP-GRAM) TRAINING, SIMILARITY & PCA PLOT
===================================================================
""")

raw_text = """king queen prince princess man woman paris france rome italy apple banana fruit animal""".split()
unique_words = list(dict.fromkeys(raw_text))
word2idx = {w: i for i, w in enumerate(unique_words)}
idx2word = {i: w for i, w in enumerate(unique_words)}
vocab_size = len(unique_words)

# Skip-gram pairs (target, context)
training_pairs = [
    ("king", "man"), ("queen", "woman"), ("prince", "man"), ("princess", "woman"),
    ("king", "queen"), ("man", "woman"), ("paris", "france"), ("rome", "italy"),
    ("apple", "fruit"), ("banana", "fruit"), ("apple", "banana"), ("king", "prince"),
    ("queen", "princess"), ("paris", "rome"), ("france", "italy")
] * 20

class SkipGramModel(nn.Module):
    def __init__(self, vocab_sz, embed_dim=16):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_sz, embed_dim)
        self.out_embed = nn.Linear(embed_dim, vocab_sz)
    def forward(self, target):
        v = self.in_embed(target)
        scores = self.out_embed(v)
        return scores

sg_model = SkipGramModel(vocab_size, embed_dim=16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sg_model.parameters(), lr=0.02)

# Train Word2Vec Skip-Gram
for epoch in range(100):
    for target_word, ctx_word in training_pairs:
        t_tensor = torch.tensor([word2idx[target_word]], dtype=torch.long)
        c_tensor = torch.tensor([word2idx[ctx_word]], dtype=torch.long)
        
        optimizer.zero_grad()
        logits = sg_model(t_tensor)
        loss = criterion(logits, c_tensor)
        loss.backward()
        optimizer.step()

# Extract learned embeddings
embeddings = sg_model.in_embed.weight.detach().numpy()

def get_nearest(word, topn=2):
    vec = embeddings[word2idx[word]].reshape(1, -1)
    sims = cosine_similarity(vec, embeddings)[0]
    sorted_ids = np.argsort(sims)[::-1]
    results = [(idx2word[i], sims[i]) for i in sorted_ids if idx2word[i] != word][:topn]
    return results

print("--- Word2Vec Nearest Neighbors ---")
for q in ["king", "paris", "apple"]:
    print(f"Nearest to '{q}': {[n[0] for n in get_nearest(q)]}")

# Vector Analogy: king - man + woman
v_king = embeddings[word2idx["king"]]
v_man = embeddings[word2idx["man"]]
v_woman = embeddings[word2idx["woman"]]
target_v = (v_king - v_man + v_woman).reshape(1, -1)
analogy_sims = cosine_similarity(target_v, embeddings)[0]
analogy_best = idx2word[np.argmax(analogy_sims)]
print(f"Vector Analogy (king - man + woman): '{analogy_best}'")

# 2D PCA Plot
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 5))
plt.scatter(coords[:, 0], coords[:, 1], color='navy', s=70)
for i, w in enumerate(unique_words):
    plt.annotate(w, xy=(coords[i, 0] + 0.02, coords[i, 1] + 0.02), fontsize=11, weight='bold')
plt.title("2D PCA of Word2Vec Embeddings", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("w2v_pca_clusters.png", dpi=300)
plt.show()

# =====================================================================
# PART C: HUGGING FACE STATIC VS CONTEXTUAL EMBEDDINGS
# =====================================================================
print("""
===================================================================
PART C: HUGGING FACE STATIC VS CONTEXTUAL EMBEDDINGS (DISTILBERT)
===================================================================
""")

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModel.from_pretrained(MODEL_NAME)
hf_model.eval()

sample_sentence = "Unprecedented advancements in NLP."
encoded_hf = tokenizer(sample_sentence, return_tensors="pt")
tokens_hf = tokenizer.convert_ids_to_tokens(encoded_hf["input_ids"][0])

print(f"Sample Sentence: '{sample_sentence}'")
print(f"WordPiece Subword Tokens: {tokens_hf}")
print(f"Input IDs: {encoded_hf['input_ids'][0].tolist()}")

# Polysemy Analysis for "bank"
sent_finance = "He deposited his money at the bank branch."
sent_nature  = "They walked along the river bank during sunset."

inputs_fin = tokenizer(sent_finance, return_tensors="pt")
inputs_nat = tokenizer(sent_nature, return_tensors="pt")

with torch.no_grad():
    out_fin = hf_model(**inputs_fin)
    out_nat = hf_model(**inputs_nat)

tokens_fin = tokenizer.convert_ids_to_tokens(inputs_fin["input_ids"][0])
tokens_nat = tokenizer.convert_ids_to_tokens(inputs_nat["input_ids"][0])

idx_bank_fin = tokens_fin.index("bank")
idx_bank_nat = tokens_nat.index("bank")

contextual_bank_fin = out_fin.last_hidden_state[0, idx_bank_fin].unsqueeze(0).numpy()
contextual_bank_nat = out_nat.last_hidden_state[0, idx_bank_nat].unsqueeze(0).numpy()

polysemy_sim = cosine_similarity(contextual_bank_fin, contextual_bank_nat)[0][0]

print(f"\nCosine Similarity of 'bank' (Finance vs Nature context): {polysemy_sim:.4f}")

# Results Table & Persistence
summary_df = pd.DataFrame([
    {"Representation": "Static (Word2Vec)", "Context-Aware": "No", "Cross-Context Similarity": "1.0000", "OOV Handling": "<UNK>"},
    {"Representation": "Contextual (DistilBERT)", "Context-Aware": "Yes", "Cross-Context Similarity": f"{polysemy_sim:.4f}", "OOV Handling": "Subwords"}
])

print("\n--- Consolidated NLP Representation Summary ---")
print(summary_df.to_string(index=False))

summary_df.to_csv("nlp_representation_summary.csv", index=False)
torch.save(sg_model.state_dict(), "custom_word2vec.pth")
print("\nSaved artifacts: 'nlp_representation_summary.csv', 'custom_word2vec.pth', and 'w2v_pca_clusters.png'!")