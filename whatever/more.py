# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
# load data
f = open('foo.csv', 'w')

print(f)

#dataset = loadtxt(f, delimiter=",")

#X = dataset[:,0:8]
#y = dataset[:,8]

#model = XGBClassifier()
#model.fit(X, y)

#print(model.feature_importances_)

#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()