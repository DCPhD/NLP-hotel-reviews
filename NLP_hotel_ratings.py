import pandas as pd
import sklearn
import re
import matplotlib.pyplot as plt

pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 2000000)

# import hotel review dataset from github
data = pd.read_csv("https://github.com/Thinkful-Ed/data-201-resources/raw/master/hotel-reviews.csv")


# select only 3 columns, make new dataframe
data = data[["name", "reviews.rating", "reviews.text"]]


# make all letters lowercase, remove all special characters with blank space except (.),
# replace NaN with blank space
data["reviews.text"] = data["reviews.text"].str.lower().replace(r"\.|\!|\?|\'|,|-|\(|\)","").fillna('')


# add column called review sentiment, transform numeric rating to string
data["review_sentiment"] = data["reviews.rating"].map({1.0: "very poor",2.0: "poor", 3.0: "Average", 4.0: "Good", 5.0: "Excellent"},)

# drop original rating column
data = data.drop(columns="reviews.rating")

# drop any NaN values in sentiment column
data = data.dropna(subset=["review_sentiment"])

# plot sentiment data
data = data["review_sentiment"].value_counts().plot(kind="bar")
plt.title("Label counts in dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show(data)

# create bag of words
from sklearn.feature_extraction.text import CountVectorizer

# limit words to 5000
vectorizer = CountVectorizer(max_features=5000)

# structure data
X = vectorizer.fit_transform(data["reviews.text"])

# put words into dataframe
bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

#print(bag_of_words.shape)

### Modeling ###

# features
X = bag_of_words

# sentiment column
y = data["review_sentiment"]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

trained_model = model.fit(X, y)

# test a self-generated review
test_review = ["I really liked the pool, and room service was superb"]

# convert test review to bag of words
X_test = vectorizer.transform(test_review).toarray()

# use model to predict label
prediction = trained_model.predict(X_test)

# predict probability
probas = trained_model.predict_proba(X_test)[0] * 100
print(probas)
# output = [3.28041235e+00 7.60875850e+01 2.04525399e+01 1.04741011e-01
#  7.47218107e-02]

# more readable option
probabilities = [str(int(x*100))+ "%" for x in probas]
labels = list(trained_model.classes_)
print(dict(zip(probabilities, labels)))
# output = {'3%': 'Average', '76%': 'Excellent', '20%': 'Good', '0%': 'very poor'}
