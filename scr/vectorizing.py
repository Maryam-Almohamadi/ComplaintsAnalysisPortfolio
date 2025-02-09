from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorizer(text, technique): # By passing the parameters "BoW" or "TF-IDF" the function apply chosen vectorization technique to text.
    vectorize = CountVectorizer(max_features=1000, stop_words='english') \
        if technique == "BoW" else TfidfVectorizer(max_features=1000, stop_words='english')
    matrix = vectorize.fit_transform(text)
    return matrix, vectorize
