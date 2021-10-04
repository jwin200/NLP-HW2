import os
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def main():
    neg_path = './Homework2-Data/neg/'
    pos_path = './Homework2-Data/pos/'
    data = {}
    i = 0
    for f in os.listdir(neg_path):
        with open(neg_path + f, 'rb') as file:
            review = str(file.read().lower())
            data[i] = [review, 'neg']
            i += 1
    for f in os.listdir(pos_path):
        with open(pos_path + f, 'rb') as file:
            review = str(file.read().lower())
            data[i] = [review, 'pos']
            i += 1
    cleaned_text = []
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Text', 'Sentiment'])
    for item in df.iterrows():
        this_item = item[1][0]  # Get each review
        tokens = nltk.WordPunctTokenizer().tokenize(this_item)
        p = re.compile(r'[^a-z]')
        new_tokens = []
        for token in tokens:
            if not p.match(token) and len(token) > 3:
                new_tokens.append(token)
        cleaned_text.append(new_tokens)
    df['Cleaned_Text'] = cleaned_text
    print(df)

    training, testing = train_test_split(df)
    Y_train = training['Sentiment'].values
    Y_test = testing['Sentiment'].values
    X_train, X_test, feature_transformer = extract_features(df, 'Text', training, testing)  # Problem here?

    reg = LogisticRegression(verbose=1, max_iter=1000)
    model = reg.fit(X_train, Y_train)
    predictions = list(reg.predict(X_test))

    correct = 0
    total = len(testing)
    print(f'Testing {total} documents...')
    i = 0
    for truth in Y_test:
        if truth == predictions[i]:
            correct += 1
        i += 1

    print(f'{round((correct / total) * 100, 2)}% accuracy')
    return 0


def extract_features(df, field, training_data, testing_data):
    """Extract features using different methods. Source code:
    https://kavita-ganesan.com/news-classifier-with-logistic-regression-in-python/"""
    # COUNT BASED FEATURE REPRESENTATION
    cv = CountVectorizer(binary=False, max_df=0.95)
    cv.fit_transform(training_data[field].values)
    train_feature_set = cv.transform(training_data[field].values)
    test_feature_set = cv.transform(testing_data[field].values)

    return train_feature_set, test_feature_set, cv


if __name__ == '__main__':
    main()
