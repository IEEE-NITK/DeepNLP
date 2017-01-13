# Overview of what will be covered

##Types of Representations of words:

1. Hypernym model
2. One hot vectors

###Problems:

1. Scalability with increase in Vocabulary
2. Synonyms
3. Human intervention - new slang words that come up.


### So, Word vectors.

## Co-Occurrence matrix

SVD
Words as dense vectors.

### Problems with SVD.
1. Again scalability.
2. Computationally Expensive.
3. Adding new words?

## Getting started with learning word vectors

We need a way to map words to vectors so that they learn from and represent the corpus.

1. Neural Probabilistic Models
    Learning through backpropagation.
    Uses a softmax to increase probability of finding next word given history.
    Assignment Provided.
    Problem: Scalability, Uses entire vocab at each training set.

2. word2vec
    CBOW/skip-gram
    Instead of creating a full probabilistic model, we turn it into a binary classification problem.
    We focus on skip-gram. CBOW is the exact opposite.

    Using tensorflow in-built helper functions to learn word embeddings.
    t-SNE(optional)
