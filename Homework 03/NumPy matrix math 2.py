# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:06:31 2019

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  
# Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of 
# decsion making problems that are currently not leveraging machine learning.

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.


#to determine random values for weights
print(np.array([np.random.dirichlet(np.ones(4),size=1)]))



people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },
          'Mary': {'willingness to travel': 0.49337138 ,
                  'desire for new experience': 0.41879654,
                  'cost': 0.05525843,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.03257365,
                  },
          'Mike': {'willingness to travel': 0.08936756,
                  'desire for new experience': 0.14813813,
                  'cost': 0.43602425,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.32647006,
                  },
          'Alice': {'willingness to travel': 0.05846052,
                  'desire for new experience': 0.6550466,
                  'cost': 0.1020457,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.18444717,
                  },
          'Skip': {'willingness to travel': 0.08534087,
                  'desire for new experience': 0.20286902,
                  'cost': 0.49978215,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.21200796,
                  },
          'Kira': {'willingness to travel': 0.14621567,
                  'desire for new experience': 0.08325185,
                  'cost': 0.59864525,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.17188723,
                  },
          'Moe': {'willingness to travel': 0.05101531,
                  'desire for new experience': 0.03976796,
                  'cost': 0.06372092,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.84549581,
                  },
          'Sara': {'willingness to travel': 0.18780828,
                  'desire for new experience': 0.59094026,
                  'cost': 0.08490399,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.13634747,
                  },
          'Tom': {'willingness to travel': 0.77606127,
                  'desire for new experience': 0.06586204,
                  'cost': 0.14484121,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01323548,
                  }                  
          }

# Transform the user data into a matrix(M_people). Keep track of column and row ids.

                                       # convert each person's values to a list

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            

#here are some lists that show column keys and values
print(peopleKeys)
print(peopleValues)



peopleMatrix = np.array(peopleValues)

peopleMatrix.shape


# Next you collected data from an internet website. You got the following information.

#1 is bad, 5 is great

np.random.randint(5, size=4)+1

restaurants  = {'flacos':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                        },
              'Joes':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Poke':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },                      
              'Sush-shi':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Chick Fillet':{'distance' : 3,
                        'novelty' : 2,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Mackie Des':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Michaels':{'distance' : 2,
                        'novelty' : 1,
                        'cost': 1,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Amaze':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Kappa':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Mu':{'distance' : 3,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      }                      
}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.


restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

#here are some lists that show column keys and values
print(restaurantsKeys)
print(restaurantsValues)

len(restaurantsValues)
#reshape to 2 rows and 4 columns

#converting lists to np.arrays is easy
restaurantsMatrix = np.reshape(restaurantsValues, (10,4))

restaurantsMatrix

restaurantsMatrix.shape

# Matrix multiplication
# Dot products are the matrix multiplication of a row vectors and column vectors (n,p) * (p,n)
#  shape example: ( 2 X 4 ) * (4 X 2) = 2 * 2
a = np.array([[1, 0], [0, 1]])
b = np.array([[1],[2]])

a.shape, b.shape

# when 2D arrays are involved, np.dot give the matrix product.
np.dot(a,b)

# documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
# intuition: https://www.mathsisfun.com/algebra/matrix-multiplying.html
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[7,8],[9,10],[11,12]])

c.shape, d.shape

np.dot(c,d)
# What is a matrix product?
# https://en.wikipedia.org/wiki/Matrix_multiplication
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
# matmul give the matrix product, too.
np.matmul(a,b)


restaurantsMatrix.shape, peopleMatrix.shape
#However, this won't work...
np.matmul(restaurantsMatrix, peopleMatrix)

# The most imporant idea in this project is the idea of a linear combination.

# Informally describe what a linear combination is and how it will relate to our resturant matrix.

    #This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
    # essentially you are multiplying each term by a constant and summing the results.

# Choose a person and compute(using a linear combination) the top restaurant for them.  
# What does each entry in the resulting vector represent?

print(peopleKeys)
print(peopleValues)

print(restaurantsKeys)
print(restaurantsValues)


restaurantsMatrix.shape, peopleMatrix.shape


#We need to swap axis on peopleMatrix
#newPeopleMatrix = np.swapaxes(peopleMatrix, 1, 0)

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

newPeopleMatrix.shape, restaurantsMatrix.shape
restaurantsMatrix.shape, newPeopleMatrix.shape

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
#Let's check our answers

results = np.matmul(restaurantsMatrix, newPeopleMatrix)

results                             


# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
# I believe that this is what John and  is asking for, sum by columns
np.sum(results, axis=1)

restaurantsKeys

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal resturant choice.
results

# Say that rank 1 is best

#reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
sortedResults = results.argsort()[::-1] +1
sortedResults

#What is the problem here? 

results.shape

#first plot heatmap
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
plot_dims = (12,10)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=results, annot=True)
plt.show()

#remember a_ij is the score for a restaurant for a person
#x is the person, y is the restaurant

print(peopleKeys)
#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice), 
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

print(restaurantsKeys)
#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)
 

#What is the problem if we want to do clustering with this matrix?


results.shape 

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


peopleMatrix.shape

#we don't need to apply standard scaler since the data is already scaled
#sc = StandardScaler()  
#peopleMatrixScaled = sc.fit_transform(peopleMatrix)  

#The example PCA was taken from.
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

print(pca.components_)
print(pca.explained_variance_)




#This function was shamefully taken from the below and modified for our purposes
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# plot principal components
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 1], ax=ax)
draw_vector([0, 0], [1, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
fig.show



# Now use peoplePCA for clustering and plotting
# https://scikit-learn.org/stable/modules/clustering.html 
kmeans = KMeans(n_clusters=3)
kmeans.fit(peopleMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c."]
labelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']

for i in range(len(peopleMatrixPcaTransform)):
   print ("coordinate:" , peopleMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()

#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)

#remember, that the order here is:

#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice), 
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)



#Now do the same for restaurants

#The example PCA was taken from.
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
restaurantsMatrix.shape

pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  

print(pca.components_)
print(pca.explained_variance_)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax)
draw_vector([0, 0], [3, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-4, 4), ylim=(-4, 4))
fig.show



# Now use restaurantsMatrixPcaTransform for plotting 
# https://scikit-learn.org/stable/modules/clustering.html
kmeans = KMeans(n_clusters=3)
kmeans.fit(restaurantsMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c."]
labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']

for i in range(len(restaurantsMatrixPcaTransform)):
   print ("coordinate:" , restaurantsMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()

#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)

#remember, that the order here is:
#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)



#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# I used "single" linkage, 
# but you could try "complete", "average", "weighted", "centroid", "median", or "ward"

pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

#Now lets try heirarchical clustering
linked = linkage(peopleMatrixPcaTransform, 'single')

#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice), 
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

labelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']

# explicit interface
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()  




#Now do the same for restaurants
pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  


#Now lets try heirarchical clustering
linked = linkage(restaurantsMatrixPcaTransform, 'single')


#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()  

