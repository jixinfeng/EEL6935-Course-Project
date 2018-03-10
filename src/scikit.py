import numpy as np
from sklearn.linear_model import LogisticRegression

VOCAB_SIZE = 89526
TRAINING_BOW = "aclImdb/train/labeledBow.feat"
TESTING_BOW = "aclImdb/test/labeledBow.feat"
NUM_TRAINING_EXAMPLES = 25000
NUM_TESTING_EXAMPLES = 25000


def process_data(filename, num_examples, class_type, encoding_type):
    class_ix = {str(rating): rating - 1 for rating in range(1,11)}

    if class_type == 'binary':
        num_classes = 2
    else:
        num_classes = 10

    # Produce x vectors that look like [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, ...]
    X = np.zeros((num_examples, VOCAB_SIZE), np.int8)
    # Produce y vectors that look like [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] or [0, 1]
    Y = np.zeros(num_examples, np.int8)

    with open(filename) as f:
        linenum = 0
        for line in f.readlines():
            tokens = line.split()
            class_, bow = tokens[0], tokens[1:]

            # Mark every word appearing in the document (movie review)
            for word in bow:
                index, count = word.split(':')
                index = int(index) - 1
                X[linenum, index] = 1 if encoding_type == 'presence' else count

            # Mark which class the document belongs to
            if class_type == 'binary':
                Y[linenum] = 0 if class_ix[class_] <= 5 else 1
            else:
                Y[linenum] = class_ix[class_]

            linenum += 1

    return X, Y



def main():
    # Create a logistic regression model,
    # and train on the Stanford Movie Review Dataset
    X, Y = process_data(TRAINING_BOW, NUM_TRAINING_EXAMPLES, 'binary', encoding_type='counts')
    print("Example input: "+str(X[:5])+str(Y[:5]))
    print("Finished loading training data")
    model = LogisticRegression()
    model.fit(X, Y)
    print("Finished training logistic regression")
    XTest, YTest = process_data(TESTING_BOW, NUM_TESTING_EXAMPLES, 'binary', encoding_type='counts')
    print("Finished loading testing data")
    print("Train accuracy:", model.score(X, Y))
    YPred = model.predict(XTest)
    correct = 0
    for i, y in enumerate(YPred):
        if y == YTest[i]:
            correct += 1
    print("Test Accuracy: "+str(float(correct)/25000))

if __name__ == "__main__":
    main()