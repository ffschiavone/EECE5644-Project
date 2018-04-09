import numpy as np
import graphviz
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# Training Data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Test Data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
tmp = clf.fit(iris.data, iris.target)
print(clf.predict(test_data))


# viz code
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)  
graph 

# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(clf,
#                      out_file=dot_data,
#                      feature_names=iris.feature_names,
#                      class_names=iris.target_names,
#                      filled=True, rounded=True,
#                      impurity=True)
# 
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")
# 
# print(test_data[0], test_target[0])
# print(iris.feature_names, iris.target_names)
#
# ############################################################
#
# # viz code from
# # http://graphviz.org/?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt
# # http://scikit-learn.org/stable/modules/tree.html#tree
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
#
# dot_data = tree.export_graphviz(clf, out_file=None,
#                      feature_names=iris.feature_names,
#                      class_names=iris.target_names,
#                      filled=True, rounded=True,
#                      special_characters=True)
# graph = graphviz.Source(dot_data)
# graph