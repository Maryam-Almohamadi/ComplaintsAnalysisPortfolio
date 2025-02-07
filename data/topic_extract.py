from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Funection to extract topics using "LDA" or "NMF"
def t_extract(texts, technique="LDA", topics_num=5): # number of topics is set to 5 for comprehension and simiplicty
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # TF-IDF convert input text to numerical matrix
    matrix = vectorizer.fit_transform(texts)

    model = (LatentDirichletAllocation if technique == "LDA" else NMF)(
        n_components=topics_num, random_state=42
    )
    model.fit(matrix) # fits chosen modelling technique to TF-IDF text data

    score = compute_coherence(model.components_, vectorizer)
    return model, model.components_, score

# Function that calculate the coherence score of the topics based on cosine similarity
def compute_coherence(topics, vectorizer, top_words_num=20):
    # Function is passed topics (numpy numerical vector), fitted CountVectorizer or TfidfVectorizer, number of top words
   
    vocab = vectorizer.get_feature_names_out()
   
    t_words = [vocab[t.argsort()[-top_words_num:][::-1]] for t in topics] 
    # This sorts topic's word distributions then selects the important words,
    # before reversing them using [::-1] to obtain descending importance rankings.

    # Binary vectors for cosine similarity
    vectors_b = [
        np.isin(vocab, words).astype(int) for words in t_words # each top word is marked 1 and the rest is 0
    ]

    # calculate cosine similarity between each topic vectors pair
    similarity = [
        cosine_similarity([vectors_b[i]], [vectors_b[j]])[0, 0] 
        for i in range(len(vectors_b))
        for j in range(i + 1, len(vectors_b))
    ]

    return np.mean(similarity) if similarity else 0 # returns pairwaise similarity average to get a coherence score