import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier


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
for n in range(1, 31, 2):
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
print preds



pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()


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
# now let's try a bagging example
clf = KNeighborsClassifier(n_neighbors=n)

bagging_clf = BaggingClassifier(
    base_estimator=clf,
    # bag using 20 trees
    n_estimators=20,
    # the max number of samples to draw from the training set for each tree
    # there are 151 training samples, so each tree will have .8 * 151
    # data points each to train on, chosen randomly with replacement
    max_samples=0.8, verbose=0
)

# use K-fold cross validation, with k=5 and get a list of accuracies
# this separates the data into training and test set 5 different times for us
# and finds out the accuracy in each case to get a sense of the average accuracy
scores = cross_val_score(bagging_clf, df[features], df["target"], cv=5)

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results
print "%r for k produced the greatest accuracy, which was %r" % (K_instance, accuracyGreatest)
print scores
# print the average accuracy
print scores.mean()
# print the standard deviation of the scores
print np.std(scores)