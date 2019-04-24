import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import datasets
#import matplotlib.pyplot as plt

# ignoring sklearn warnings
from warnings import filterwarnings
filterwarnings("ignore")

# adapt this code below to run your analysis

# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

n_folds = 6 #just pick a number i guess

#data
iris = datasets.load_iris()

iris_data = (iris.data, iris.target, n_folds)

#classifiers
classifiers = {RandomForestClassifier: [{'min_samples_split': 2,
                                         'min_samples_leaf': 1,
                                         'max_features': 'auto',
                                         },
                                         {'min_samples_split': 4,
                                          'min_samples_leaf': 2,
                                          'max_features': 'auto',
                                         },
                                         {'min_samples_split': 2,
                                          'min_samples_leaf': 1,
                                          'max_features': 'log2'
                                         },
                                        ]
              }

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist

    clf.fit(M[train_index], L[train_index])

    pred = clf.predict(M[test_index])

    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

def multi_classification(clfs, data):
    if type(clfs) == list:
        clfs = {clfs: [{}] for clf in clfs}

    ret = []

    for clf in clfs:
        hps = clfs[clf]

        for hp in hps:
            ret.append(run(clf, data, hp))

    return ret

#results = run(RandomForestClassifier, iris_data, clf_hyper={})

results = multi_classification(classifiers, iris_data)

#for i in results:
#    print(results[i]['train_index'])

print(results)

with open('out.txt', 'w') as f:
    for item in results:
        f.write("%s\n" % item)
