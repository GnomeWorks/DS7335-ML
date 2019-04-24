#!/usr/bin/python
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc as auc_func, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance
from collections import OrderedDict
from itertools import cycle

# adapt this code below to run your analysis

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

class GridDefault(object):
    def __init__(self):
        self.modelname = "n/a"
        self.modeltype = None
        self.defaults = {
            'random_state': 0xdedede,
            'default_2' : 'filler'
        }
        self.parameters = ["p1", 'p2']
        self.tuning_options = {
            'p1': ['o1', 'o2'],
            'p2': np.linspace(0, 1, 20)
        }

        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]

        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

    def fit_and_predict(self, x_train, y_train, x_test, y_test):
        model_params = { **self.defaults, **{p: self.tuning_state[p] for p in self.parameters}}
        model = self.modeltype(**model_params)
        model.fit(x_train, y_train)
        return (model.predict_proba(x_test)[:,1], model.predict(x_test), y_test)

class GridLogisticReg(GridDefault):
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = "LogisticRegression"
        self.modeltype = LogisticRegression
        self.defaults = {
            'random_state': 0xdedede,
            'n_jobs': 1
        }
        self.parameters = ['C', 'penalty']
        self.tuning_options = {
            'C': np.linspace(.01, 1, 20),
            'penalty': ['l1', 'l2']
        }

        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]

        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridDecisionTree(GridDefault):
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = "DecisionTreeClassifier"
        self.modeltype = DecisionTreeClassifier
        self.defaults = {
            'random_state': 0xdedede
        }
        self.parameters = [
            "min_weight_fraction_leaf",
            'max_depth'
            ]
        self.tuning_options = {
            'min_weight_fraction_leaf': np.linspace(0, .4, 15),
            'max_depth': np.arange(3, 15, 1)
        }

        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]

        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridRandomForest(GridDefault):
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = 'RandomForestClassifier'
        self.modeltype = RandomForestClassifier
        self.defaults = {
            'random_state' : 0xdedede,
            'min_samples_leaf': 1,
            'n_jobs' : 8
        }
        self.parameters = [
            'n_estimators',
            'max_depth'
            ]
        self.tuning_options = {
            'n_estimators': np.arange(4, 20, 3),
            'max_depth': np.arange(1, 20, 1)
        }

        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]

        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridXGBoost(GridDefault):
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = 'XGBoostClassifier'
        self.modeltype = XGBClassifier
        self.defaults = {
            'random_state' : 0xdedede,
            'n_jobs' : 8
        }
        self.parameters = [
            'n_estimators',
            'max_depth'
            ]
        self.tuning_options = {
            'n_estimators': np.arange(4, 20, 3),
            'max_depth': np.arange(1, 20, 1)
        }

        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]

        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridSearch(object):
    def __init__(self, data, labels, data_cols = None, label_name = 'y', numfolds = 5, output_folder = os.getcwd()):
        if not data_cols:
            data_cols = np.arange(data.shape[1])
        self.data_cols = data_cols
        self.data = data
        self.labels = labels
        self.label_names = label_name
        self.numfolds = numfolds
        self.output_folder = output_folder
        self.colors = cycle(['turquoise', 'dodgerblue', 'slategrey', 'skyblue', 'teal', 'cornflowerblue', 'lavender', 'blue'])
		
        folds = np.random.random(self.data.shape[0])
        
        self.train_folds_X = [data[(folds <= i) | (folds > i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.train_folds_y = [labels[(folds <= i) | (folds > i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.test_folds_X = [data[(folds > i) & (folds <= i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.test_folds_y = [labels[(folds > i) & (folds <= i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.train_test_folds = tuple(zip(
            self.train_folds_X,
            self.train_folds_y,
            self.test_folds_X,
            self.test_folds_y
        ))
        
        self.models = {
            'lreg': GridLogisticReg(),
            'dtree': GridDecisionTree(),
            'rforest': GridRandomForest(),
            'xgboost': GridXGBoost()
        }
        
        self.results_auc = {
          model_name :
              np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters])).reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
              for model_name, model_i in self.models.items()
          }
        
        self.results_precision = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters])).reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        
        self.results_recall = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters])).reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        
        self.results_accuracy = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters])).reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        
        if 'data' not in os.listdir(output_folder):
            os.mkdir('data')
        
        if 'charts' not in os.listdir(output_folder):
            os.mkdir('charts')

    def optimize_all_models(self):
        for m_name in self.models.keys():
            self.optimize_model(m_name)

    def output_auc_plot(self, model_idx, results):
        plt.figure(figsize=(10, 10))
        plt.title('ROC (Receiver Operating Characteristic)')
        sum_aucs = 0

        for fold in range(self.numfolds):
            probs, _, y_te = results[fold]
            fpr, tpr, _ = roc_curve(y_te, probs)
            roc_auc = auc_func(fpr, tpr)
            plt.plot(fpr, tpr, color = next(self.colors), label = 'AUC = %0.3f' % roc_auc)
            sum_aucs += roc_auc

        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        modi = self.models[model_idx]
        file_name = self.output_folder + "\\charts\\" + modi.modelname + ".".join([str(p) + "-" +  str(v) for p,v in modi.tuning_state.items()]) + "__auc-{:.3f}.png".format(sum_aucs / self.numfolds)
        plt.savefig(file_name)
        plt.close()

    def save_model_data(self, model_idx, results):
        aucs = []
        precs = []
        recs = []
        accs = []
        for y_probs, y_pred, y_te in results:
            mtrx = confusion_matrix(y_te, y_pred)
            tp = mtrx[0,0]
            fp = mtrx[0,1]
            fn = mtrx[1,0]
            fpr, tpr, _ = roc_curve(y_te, y_probs)
            aucs.append(auc_func(fpr, tpr))
            precs.append( float(tp) / (tp + fp) )
            recs.append( float(tp) / (tp + fn) )
            accs.append( float((y_pred == y_te).sum()) / y_te.shape[0] )
        auc = sum(aucs) / len(aucs)
        prec = sum(accs) / len(accs)
        rec = sum(recs) / len(recs)
        accs = sum(accs) / len(accs)
        modi = self.models[model_idx]

        ## Area under ROC curve
        self.results_auc[model_idx][modi.numeric_state[0], modi.numeric_state[1]] = auc            
        
		## Precision
        self.results_precision[model_idx][modi.numeric_state[0], modi.numeric_state[1]] = prec
		
        ## Recall
        self.results_recall[model_idx][modi.numeric_state[0], modi.numeric_state[1]] = rec
		
        ## Accuracy
        self.results_accuracy[model_idx][modi.numeric_state[0], modi.numeric_state[1]] = accs

        return auc

    def diff_measure(self, model_idx, param, p_int, min_opts=8):
        modi = self.models[model_idx]
        aucs = OrderedDict()
        settings = modi.tuning_options[param]
        self.models[model_idx].tuning_state[param] = settings[0]
        aucs[settings[0]] = self.run_one_model(model_idx)
		
        for idx in np.arange(len(settings) / min_opts, len(settings), len(settings)/ min_opts):
            self.models[model_idx].tuning_state[param] = settings[int(idx)]
            self.models[model_idx].numeric_state[p_int] = int(idx)
            aucs[settings[int(idx)]] = self.run_one_model(model_idx)
        
        self.models[model_idx].tuning_state[param] = settings[-1]
        aucs[settings[-1]] = self.run_one_model(model_idx)
        vals = tuple(aucs.values())
        keys = tuple(aucs.keys())
        
        for auc_idx in range(1, len(vals))[:-1]:
            if (vals[auc_idx] > vals[auc_idx -1]) and (vals[auc_idx] > vals[auc_idx + 1]):
                for s_int, sett in enumerate(settings[[i for i,v in enumerate(settings) if v == keys[auc_idx]][0]:
                                       [i for i,v in enumerate(settings) if v == keys[auc_idx + 1]][0]]):
                    self.models[model_idx].tuning_state[param] = sett
                    self.models[model_idx].numeric_state[p_int] = s_int
                    aucs[int(idx)] = self.run_one_model(model_idx)

    def split_param(self, model_idx, param, min_params=10):
        jump = min_params - 2
        settings = self.models[model_idx].tuning_options[param]
        
        if len(settings) >= min_params:
            idxs = np.linspace(0, len(settings)-1, min_params).astype(int)
            vals = [settings[idx] for idx in idxs]
            return zip(idxs, vals)
        else:
            return enumerate(settings)

    def fill_missing(self, model_idx, row_col, num, prev, nxt, row = True):
        p_0 = self.models[model_idx].parameters[0]
        p_1 = self.models[model_idx].parameters[1]
        s_0 = self.models[model_idx].tuning_options[p_0]
        s_1 = self.models[model_idx].tuning_options[p_1]

        row = self.models[model_idx].numeric_state[0]
        col = self.models[model_idx].numeric_state[1]

        auc_mtrx = self.results_auc[model_idx]
        dummy_mtrx = np.arange(auc_mtrx.size).reshape(*auc_mtrx.shape)
        
        idx_min = dummy_mtrx[auc_mtrx == prev][0]
        idx_min_col = idx_min % auc_mtrx.shape[0]
        idx_min_row = int(idx_min / auc_mtrx.shape[1])

        idx_max = dummy_mtrx[auc_mtrx == nxt][0]
        idx_max_col = idx_max % auc_mtrx.shape[0]
        idx_max_row = int(idx_max / auc_mtrx.shape[1])

        if row_col == 'row':
            self.models[model_idx].tuning_state[p_0] = s_0[num]
            for col in range(idx_min_row + 1, idx_max_row - 1):
                self.models[model_idx].numeric_state = [num, col]
                self.models[model_idx].tuning_state[p_1] = s_1[col]
                self.run_one_model(model_idx)
        elif row_col == 'col':
            self.models[model_idx].tuning_state[p_1] = s_1[num]
            for row in range(idx_min_col + 1, idx_max_col - 1):
                self.models[model_idx].numeric_state = [row, num]
                self.models[model_idx].tuning_state[p_0] = s_0[row]
                self.run_one_model(model_idx)

    def optimize_model(self, model_idx, min_params=10):
        modi = self.models[model_idx]
        s_y = list(self.split_param(model_idx, modi.parameters[0]))
        p_y = modi.parameters[0]
        s_x = list(self.split_param(model_idx, modi.parameters[1]))
        p_x = modi.parameters[1]
        for y_idx, y_val in s_y:
          self.models[model_idx].tuning_state[p_y] = y_val
          self.models[model_idx].numeric_state[0] = y_idx
          for x_idx, x_val in s_x:
              self.models[model_idx].tuning_state[p_x] = x_val
              self.models[model_idx].numeric_state[1] = x_idx
              self.run_one_model(model_idx)

        auc_mtrx = self.results_auc[model_idx]
        mtrx = auc_mtrx[auc_mtrx > 0].reshape(len(s_y), len(s_x))
        for r in range(1,mtrx.shape[0] - 1):
            for c in range(1, mtrx.shape[1] - 1):
                if mtrx[r-1, c] > mtrx[r, c] and mtrx[r, c] < mtrx[r + 1, c]:
                    self.fill_missing(model_idx, 'col', c, mtrx[r-1, c], mtrx[r + 1, c])
                if mtrx[r, c-1] > mtrx[r, c] and mtrx[r, c] < mtrx[r, c + 1]:
                    self.fill_missing(model_idx, 'row', r, mtrx[r, c-1], mtrx[r, c + 1])

    def run_one_model(self, model_idx):
        results = []
        for X_tr, y_tr, X_te, y_te in self.train_test_folds:
            results.append(self.models[model_idx].fit_and_predict(X_tr, y_tr, X_te, y_te))

        auc = self.save_model_data(model_idx, results)
        self.output_auc_plot(model_idx, results)
        return auc		

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    gs = GridSearch(X, y)
    gs.optimize_all_models()
