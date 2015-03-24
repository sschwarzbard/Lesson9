import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/iris.csv")

# This creates an array of Trues and False, uniformly distributed such that
# around 30% of the items will be True and the rest will be False
test_idx = np.random.uniform(0, 1, len(df)) <= 0.3

# The training set will be ~30% of the data
train = df[test_idx==True]
# The test set will be the remaining, ~70% of the data
test = df[test_idx==False]


features = ['sepal_length','sepal_width','petal_length','petal_width']
K_instance = 0
accuracyGreatest = 0
results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 45, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[features], train['target'])
    # then make the predictions
    preds = clf.predict(test[features])
    # very simple and terse line of code that will check the accuracy
    # documentation on what np.where does: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    # Here is a simple example: suppose our predictions where [True, False, True] and the correct values were [True, True, True]
    # The next line says, create an array where when the prediction = correct value, the value is 1, and if not the value is 0.
    # So the np.where would, in this example, produce [1, 0, 1] which would be summed to be 2 and then divided by 3.0 to get 66% accuracy
    accuracy = np.where(preds==test['target'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
    results.append([n, accuracy])
    if accuracyGreatest < accuracy:
        accuracyGreatest = accuracy
        K_instance = n
results = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

# ****** Now, let's see how accurate the predictor is ******
results = []
# let's try two different weighting schemes, one where we don't worry about the distance
# another where we weight each point by 1/distance
for w in ['uniform', 'distance']:
    clf = KNeighborsClassifier(3, weights=w)
    w = str(w)
    clf.fit(train[features], train['target'])
    preds = clf.predict(test[features])

    # For an explanation of this line, refer to my explanation of this same line above
    accuracy = np.where(preds==test['target'], 1, 0).sum() / float(len(test))
    if accuracyGreatest < accuracy:
        accuracyGreatest = accuracy
        K_instance = n
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)

    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results
print "%r for k produced the greatest accuracy, which was %r" % (K_instance, accuracyGreatest)