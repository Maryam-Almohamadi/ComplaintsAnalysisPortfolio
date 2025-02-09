import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from cleaning_preprocess import cleaning
from vectorizing import vectorizer
from topic_extract import t_extract
from visuals import freq_words, coherence_plot, vectors_comparison

# Loading the csv file of financial complaints
frame = pd.read_csv(
    r"C:\Users\marya\OneDrive\Desktop\University\Ongoing\Project_Data Analysis\Data set\(1000)Consumer_Finance_Complaints.csv",
    usecols=["Consumer complaint narrative"] # loading the column utilized for analysis
)

# Preprocessing stage
sample_frame = frame.dropna().sample(n=200, random_state=42) # sampling 200 rows of the chosen column and handling missing values
sample_frame["cleaning"] = sample_frame["Consumer complaint narrative"].astype(str).apply(cleaning) # cleaning column's text

# Using the vectorization (vectorizer) method & passing arguments for each technique (Tf-IDF, BoW) respectivly
matrix_tfidf, vectorize_tfidf = vectorizer(sample_frame["cleaning"], technique="TF-IDF")
matrix_bow, vectorize_bow = vectorizer(sample_frame["cleaning"], technique="BoW")

# Using the topic extraction (t_extract) method & passing arguments for each technique (LDA, NMF) respectivly
lda_model, lda_topics, lda_coherence = t_extract(sample_frame["cleaning"], technique="LDA")
nmf_model, nmf_topics, nmf_coherence = t_extract(sample_frame["cleaning"], technique="NMF")

# Plotting graphs to visulas & compare methods applied for analysis
freq_words(vectorize_tfidf, matrix_tfidf, title="Most Frequent Words (TF-IDF)") # plotting graph showing word weights based on TF-IDF
freq_words(vectorize_bow, matrix_bow, title="Most Frequent Words (BoW)") # plotting graph showing word numerical representation based on BoW
vectors_comparison(vectorize_tfidf, matrix_tfidf, vectorize_bow, matrix_bow) # this graph comapres BoW & TF-IDF techniques
coherence_plot(["LDA", "NMF"], [lda_coherence, nmf_coherence]) # plotting coherence score for both NMF & LDA for comparison

# Function that convert LDA & NMF numerical output into human-readable top-words for each top 10 topics
def print_t(topics, vectorizer, top_n=10):
    vocab = vectorizer.get_feature_names_out()
    for IDX, topic in enumerate(topics):
        top_words_ind = topic.argsort()[-top_n:][::-1]
        top_words = [vocab[i] for i in top_words_ind] 
        print(f"Topic {IDX}: {top_words}") 

# Displaying topics and their top words for both models in the terminal
print("\nLDA Topics:")
print_t(lda_topics, vectorize_tfidf)

print("\nNMF Topics:")
print_t(nmf_topics, vectorize_tfidf)
