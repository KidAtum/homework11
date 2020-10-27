# Lucas Weakland
# Homework 11
# Decision Trees
# Tree 2 - Text

# imports
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)

# show / print
text_representation = tree.export_text(clf)
print(text_representation)



