import nltk
import json
import random


def main():
    with open('data.json', 'r') as file:
        data = json.load(file)['data']
    # Shuffle data and split into training/testing sets
    random.shuffle(data)
    training = data[0:int(len(data) / 2)]
    testing = data[int(len(data) / 2):]

    # Train Naive Bayes
    classifier = nltk.NaiveBayesClassifier.train(training)
    classifier.show_most_informative_features()
    correct = 0
    total = len(testing)
    print(f'Testing {total} documents...')

    # Classify all testing documents
    for doc in testing:
        d = doc[0]
        true_sent = doc[1]
        algo_sent = classifier.classify(d)
        # Determine if correct
        if algo_sent == true_sent:
            correct += 1

    print(f'{round((correct / total) * 100, 2)}% accuracy')
    return 0


if __name__ == '__main__':
    main()
