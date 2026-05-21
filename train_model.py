import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("text_labels.txt")

X = df['text']
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier',MultinomialNB())
])

model.fit(X_train, y_train)

pickle.dump(model, open("news_model.pkl", "wb"))

print("Model trained and saved successfully!")