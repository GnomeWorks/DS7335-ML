# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:20:48 2019

@author: GnomeWorks
"""

import functools
import io
import numpy as np
import sys
import numpy.lib.recfunctions as rfn
import time
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.model_selection import KFold  #EDIT: I had to import KFold 
from numpy import genfromtxt

import matplotlib.pyplot as plt

#if sys.version_info >= (3,):
#    np.genfromtxt = genfromtxt_py3_fixed
  
#Read the two first two lines of the file.
with open('data\claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())
    
names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]

sys.setrecursionlimit(5500)

types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

CLAIMS = genfromtxt('data\claim.sample.csv', delimiter=',', names=True, dtype=types, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

#print(CLAIMS)

test = 'J'
test = test.encode()

JcodeIndexes = np.flatnonzero(np.core.defchararray.startswith(CLAIMS['ProcedureCode'], test, start=0, end=None)!=-1)

JcodeINDICES = np.core.defchararray.startswith(CLAIMS['ProcedureCode'], test, start=1, end=2)
Jcodes = CLAIMS[JcodeINDICES]

print(Jcodes)

print(Jcodes.dtype.names)

#Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')


# Reverse the sorted Jcodes
Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]

# What are the top five J-codes based on the payment to providers?

# We still need to group the data
print(Sorted_Jcodes[:10])

# You can subset it...
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes = Sorted_Jcodes['ProcedureCode']

#recall their data types
Jcodes.dtype
ProviderPayments.dtype

Jcodes[:3]

ProviderPayments[:3]



#Join arrays together
arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

# What does the result look like?
print(Jcodes_with_ProviderPayments[:3])

Jcodes_with_ProviderPayments.shape




#http://esantorella.com/2016/06/16/groupby/
#A fast GroupBy class
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result


#See how long the groupby takes
start = time.clock()
#grouped = Groupby(Jcodes)

#perform the groupby to get the group sums
group_sums = Groupby(Jcodes).apply(np.sum, ProviderPayments, broadcast=False)
print('time to compute group sums once with Grouped: {0}'\
      .format(round(time.clock() - start, 3)))


group_sums.shape

np.set_printoptions(threshold=500, suppress=True)
print(group_sums)

#How do we get the JCodes for the group sums?
#Look up at the class Groupby
unique_keys, indices = np.unique(Jcodes, return_inverse = True)

print(unique_keys)
print(indices)

len(unique_keys)
len(group_sums)    

print(group_sums)

#Zip it and sort it.
zipped = zip(unique_keys, group_sums)  # python 3
sorted_group_sums = sorted(zipped, key=lambda x: x[1])

print(sorted_group_sums)

print(Sorted_Jcodes.dtype.names)

##We need to come up with labels for paid and unpaid Jcodes

## find unpaid row indexes  

unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)

## find paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)


#Here are our
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]

Paid_Jcodes = Sorted_Jcodes[paid_mask]

#These are still structured numpy arrays
print(Unpaid_Jcodes.dtype.names)
print(Unpaid_Jcodes[0])

print(Paid_Jcodes.dtype.names)
print(Paid_Jcodes[0])

#Now I need to create labels


print(Paid_Jcodes.dtype.descr)
print(Unpaid_Jcodes.dtype.descr)

#create a new column and data type for both structured arrays
new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

print(new_dtype1)
print(new_dtype2)

#create new structured array with labels

#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype2)

#check the shape
Unpaid_Jcodes_w_L.shape
Paid_Jcodes_w_L.shape

#Look at the data
print(Unpaid_Jcodes_w_L)
print(Paid_Jcodes_w_L)



#copy the data
Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

#And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

#Look at the data..
print(Unpaid_Jcodes_w_L)


# Do the same for the Paid set.

#copy the data
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0

#Look at the data..
print(Paid_Jcodes_w_L)


#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)

#check the shape
Jcodes_w_L.shape

#44961 + 6068

#look at the transition between the rows around row 44961
print(Jcodes_w_L[44960:44963])

#We need to shuffle the rows before using classifers in sklearn

Jcodes_w_L.dtype.names



#shuffle the rows

# Shuffle example:
    
    
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]


data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

#shuffle rows
np.random.shuffle(data)

#notice that data is still in the right order
print(data)

# We want to do the same for our data since we have combined unpaid and paid together, in that order. 

print(Jcodes_w_L[44957:44965])

# Apply the random shuffle
np.random.shuffle(Jcodes_w_L)

print(Jcodes_w_L[44957:44965])

#Columns are still in the right order
Jcodes_w_L

#Now get in the form for sklearn

Jcodes_w_L.dtype.names


# recall the features names:
#features = ['V1', 'ClaimNumber', 'ClaimLineNumber', 'MemberID', 'ProviderID',
#     'LineOfBusinessID', 'RevenueCode', 'ServiceCode', 'PlaceOfServiceCode',
#     'ProcedureCode', 'DiagnosisCode', 'ClaimChargeAmount', 'DenialReasonCode',
#     'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 'PricingIndex',
#     'CapitationIndex', 'SubscriberPaymentAmount', 'ProviderPaymentAmount',
#     'GroupIndex', 'SubscriberIndex', 'SubgroupIndex', 'ClaimType',
#     'ClaimSubscriberType', 'ClaimPrePrinceIndex', 'ClaimCurrentStatus',
#     'NetworkID', 'AgreementID']

label =  'IsUnpaid'

cat_features = ['V1', 'ProviderID','LineOfBusinessID','RevenueCode',
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DiagnosisCode', 'DenialReasonCode',
                'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType', ]
numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']

#M = Jcodes_w_L[features].copy()
#M = M.view((float, len(M.dtype.names)))

#from sklearn import preprocessing



#convert features to list, then to np.array 
# This step is important for sklearn to use the data from the structured NumPy array

#separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())

# first use Sklearn's LabelEncoder function ... then use the OneHotEncoder function
# https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# http://www.stephacking.com/encode-categorical-data-labelencoder-onehotencoder-python/

# Some claim you can do OnehotEncoder without a label encoder, but I haven't seen it work.
# https://stackoverflow.com/questions/48929124/scikit-learn-how-to-compose-labelencoder-and-onehotencoder-with-a-pipeline

# Run the Label encoder
le = preprocessing.LabelEncoder()
for i in range(20):
   Mcat[:,i] = le.fit_transform(Mcat[:,i])

# Run the OneHotEncoder
# Could encounter a memory error here in which case, you probably should subset.
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)

#What is the shape of the matrix categorical columns that were OneHotEncoded?   
Mcat.shape
Mnum.shape


#I am subsetting them since I was having memory issues.
#You might be able to decide which features are useful and remove some of them before
# the label encoder and one hot encoding step

#If you want to recover from the memory error then subset
#Mcat = np.array(Jcodes_w_L[cat_features].tolist())

Mcat_subset = Mcat[0:10000]
Mcat_subset.shape

Mnum_subset = Mnum[0:10000]
Mnum_subset.shape

L_subset = L[0:10000]

# Uncomment if you need to run again from a subset.

## Run the Label encoder
le = preprocessing.LabelEncoder()
for i in range(20):
   Mcat[:,i] = le.fit_transform(Mcat[:,i])

# Run the OneHotEncoder
# Could encounter a memory error here in which case, you probably should subset.
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)


#What is the size in megabytes before subsetting?
# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-33.php
# and using base2 (binary conversion), https://www.gbmb.org/bytes-to-mb
print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

#What is the size in megabytes after subsetting?
print("%d Megabytes" % ((Mcat_subset.size * Mcat_subset.itemsize)/1048576)) 
print("%d Megabytes" % ((Mnum_subset.size * Mnum_subset.itemsize)/1048576))


M = np.concatenate((Mcat, Mnum), axis=1)



#Concatenate the columns
M = np.concatenate((Mcat_subset, Mnum_subset), axis=1)


L = Jcodes_w_L[label].astype(int)

# Match the label rows to the subset matrix rows.
L = L[0:10000]

M.shape
L.shape

# Now you can use your DeathToGridsearch code.


n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)





#EDIT: A function, "run", to run all our classifiers against our data.

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    clf = a_clf(**clf_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
                                                                             #      for those corresponding to a formal parameter
                                                                             #      in a dictionary.
            
    clf.fit(M[train_index], L[train_index])   #EDIT: First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = clf.predict(M[test_index])         #EDIT: Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret




def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                         #We want to prevent unique k1 values due to different "key" values
                         #when we actually have the same classifer and hyper parameter settings.
                         #So, we convert to a string
                        
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
            

def myHyperSetSearch(clfsList,clfDict):
    #hyperSet = {}
    for clf in clfsList:
    
    #I need to check if values in clfsList are in clfDict
        clfString = str(clf)
        #print("clf: ", clfString)
        
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
            #Nothing to do here, we need to get into the inner nested dictionary.
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperSet = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperSet) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results
 



clfsList = [RandomForestClassifier] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]}}#,
                                      
           #'LogisticRegression': {"tol": [0.001,0.01,0.1]}}

                   
#Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

#Run myHyperSetSearch
myHyperSetSearch(clfsList,clfDict)    

print(clfsAccuracyDict)


# for determining maximum frequency (# of kfolds) for histogram y-axis
n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

# initialize the plot_num counter for incrementing in the loop below
plot_num = 1 

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
               


#create the histograms
for k1, v1 in clfsAccuracyDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
    fig = plt.figure(figsize =(20,10)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=30) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

    # pass in subplot adjustments from above.
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num) #convert plot number to string
    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    plot_num = plot_num+1 # increment the plot_num counter by 1
    
plt.show()

