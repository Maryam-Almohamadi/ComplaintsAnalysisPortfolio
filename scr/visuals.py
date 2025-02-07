import numpy as np
import matplotlib.pyplot as plt

# Plots graph of top 20 frequent words for both "BoW" & "TF-IDF"
def freq_words(vectorize, matrix, top_n=20, title="Most Frequent Words"):
    words_cnt = np.asarray(matrix.sum(axis=0)).flatten() # find the sum of the word's frequenceies
    words_ind = np.argsort(words_cnt)[-top_n:] # get the indinces of the top words

    plt.figure(figsize=(20, 10))
    plt.barh([vectorize.get_feature_names_out()[i] for i in words_ind], words_cnt[words_ind])
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title(title)
    plt.show()

# Plots the coherence score for both "LDA" & "NMF" 
def coherence_plot(models, coherence_scores): 
    plt.figure(figsize=(20, 10))
    plt.bar(models, coherence_scores, color= ["lightgreen", "deepskyblue"])
    plt.xlabel("Model")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score Comparison Between LDA & NMF")
    plt.show()

# Plots a graph comparing BoW and TF-IDF in terms of word frequncy for random words
def vectors_comparison(vectorizer_tfidf, matrix_tfidf, vectorizer_bow, matrix_bow, top_n=20):
    words_tfidf = np.asarray(matrix_tfidf.sum(axis=0)).flatten() # TF-IDF gives the weighted importance value.
    words_bow = np.asarray(matrix_bow.sum(axis=0)).flatten() # BoW gives the basic word occurrence numbers

    # selects words that exist in both vocabularies.
    common_w = set(vectorizer_tfidf.get_feature_names_out()) & set(vectorizer_bow.get_feature_names_out())

    # identify the index positions of common words found in feature matrices, used to access word frequencies
    common_ind_tfidf = [list(vectorizer_tfidf.get_feature_names_out()).index(word) for word in common_w]
    common_ind_bow = [list(vectorizer_bow.get_feature_names_out()).index(word) for word in common_w]
    
    # plotting words frequency bar for each technique
    plt.figure(figsize=(20, 10)) 
    plt.barh(list(common_w)[:top_n], words_tfidf[common_ind_tfidf][:top_n], color='red', label='TF-IDF') 
    plt.barh(list(common_w)[:top_n], words_bow[common_ind_bow][:top_n], color='orange', alpha=0.5, label='BoW')
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("TF-IDF vs BoW Frequency Comparison")
    plt.legend()
    plt.show()