from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix
