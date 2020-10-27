# Lucas Weakland
# Homework 11
# Decision Trees
# Tree 1 - PLT / Image

# imports
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load data
iris = datasets.load_iris()

X = iris.data[:, 2:]

y = iris.target

# justify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

clf_tree.fit(X_train, y_train)


fig, ax = plt.subplots(figsize=(10, 10))

tree.plot_tree(clf_tree, fontsize=10)

# show the plt
plt.show()
