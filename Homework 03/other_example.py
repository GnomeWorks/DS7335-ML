#import time
#import warnings
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
#from sklearn import cluster
from sklearn.cluster import KMeans #, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score

import sys
orig_stdout = sys.stdout
f = open('MuthuPalaniHomeWork3.txt', 'w')
sys.stdout = f

def make_dict(names, categories, floatRint):
    
    # Create People & Restaurant Dictionary with random integer (or) float
    # Sum of Category weight should be 1 for People Matrix
    # floatRInt indicator will instruct the function to execute dirichlet (or)randint function
    
    types = {}
    
    for n in names:
        if floatRint == "F":
            random_prefs = np.random.dirichlet(np.ones(len(categories)),size=1)
            # Need to reshape as dirichlet creates list of list
            random_prefs = random_prefs.reshape(len(categories),)
            random_prefs = np.around(random_prefs, decimals = 2)
        elif floatRint == "I":
            random_prefs = np.random.randint(1, 11, len(categories))
        else:
            print ("Invalid Param .. Kaboom..")
            break
        dict_input = dict(zip(categories, random_prefs))
        types[n] = dict_input
    return types


def convert_list_to_matrix(list):
    # Loop through the dictionary and create matrix
    # p_names/r_names & p_cats/r_cats are the row and column labels
    
    list_of_lists = []
    for names, choices in list.items():
        passing_list = []
        for choice in choices:
            try:
                passing_list.append(choices[choice])
            except KeyError:
                passing_list.append(0)
        list_of_lists.append(passing_list)

    passing_list_array = np.array(list_of_lists)
    return(passing_list_array)
    
def DataProcessing(people, restaurants, names):
   
    # One Stop function to create people x restaurant matrix and the rank..
    new_people = np.swapaxes(people, 0, 1)
    people_X_restaurants = np.dot(restaurants, new_people)
    restaurants_sum = np.sum(people_X_restaurants, axis=1)
    ranks = rankdata(restaurants_sum, method='max').reshape(restaurants_sum.shape)
    rankMatrix = createRankMatrix(names, ranks)
    usr_x_rest_rank = sorted(rankMatrix.items(), key=lambda kv: kv[1])

    return (people_X_restaurants, rankMatrix, usr_x_rest_rank)

def createRankMatrix(names, ranks):
    
    # Create rank from raw score.
    rankMatrix = {}
    for i in range(len(ranks)):
        name = names[i]
        rank = ranks[i]
        rankMatrix[name] = rank
    return rankMatrix

def dentogram(fitMatrix, label, title):
    pca = PCA(n_components=2)  
    PcaTransform = pca.fit_transform(fitMatrix)  

    linked = linkage(PcaTransform, 'single')
    fig = plt.figure(figsize=(17, 7))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(linked,  
               orientation='top',
               labels=label,
               distance_sort='descending',
               show_leaf_counts=True, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)
    plt.show()  
    
    return

def knn2to4(fitMatrix,subtitle):
    n_clusters = 1 # Initialize, gets incremented inside the loop.
    colors = ['#4EACC5', '#FF9C34', '#4E9A06','#377eb8'] # colors for the plot.

    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    plt.suptitle(subtitle)

    for i in range(3):
        
        ax = fig.add_subplot(1, 3, n_clusters)
        n_clusters = n_clusters+1
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fitMatrix)
        k_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(fitMatrix, k_means_cluster_centers)

        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(fitMatrix[my_members, 0], fitMatrix[my_members, 1], 'w',
                    markerfacecolor=col, marker='.', markersize=12)
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=15)
            
            ax.set_title('KMeans - %i' %n_clusters)
            ax.set_xticks(())
            ax.set_yticks(())

    plt.show()
    plt.close()

    return

# Code Reference
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def chkMaxClusters(fitMatrix): 
    
    range_n_clusters = [2, 3, 4]
    print("")
    print("Silhouette scores are between -1 to +1")
    print("-1: Incorrect, 0: Overlapping  & +1: Highly densed Clustering")
    print("Purity of the cluster is measured with higher score which also means well separated and highly densed")
    print("")
        
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(fitMatrix) + (n_clusters + 1) * 10])
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(fitMatrix)
        silhouette_avg = silhouette_score(fitMatrix, cluster_labels)
        sample_silhouette_values = metrics.silhouette_samples(fitMatrix, cluster_labels)
        
        print("For n_clusters =", n_clusters,
              "\n The average silhouette_score is :", silhouette_avg,
              "\n Individual silhouette scores were:", sample_silhouette_values,
              "\n and their assigned clusters:", cluster_labels)
        print("")
        
        sample_silhouette_values = silhouette_samples(fitMatrix, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(fitMatrix[:, 0], fitMatrix[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
            
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
                
                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")
                
                plt.suptitle(("Silhouette analysis for KMeans clustering "
                              "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        
    plt.show()
    return
  
def checkClusterPurity(fitMatrix):
    
    range_n_clusters = [2, 3, 4]
    print("Davies Bouldin Score: Lower means better separated, zero is the lowest")
    print("Calinski Harabaz Score: Higher means dense and well separated")
    print("")
    print("No. of Clusters\t\tCalinski Harabaz Index\t\tDavies Bouldin Score")
    for n_clusters in range_n_clusters:
        
        pca = PCA(n_components=n_clusters)
        mPeopleXRestaurantsPcaTransform = pca.fit_transform(fitMatrix) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mPeopleXRestaurantsPcaTransform)
        k_means_cluster_labels = kmeans.predict(mPeopleXRestaurantsPcaTransform)
        print("\t", n_clusters, 
              "\t\t", metrics.calinski_harabaz_score(mPeopleXRestaurantsPcaTransform, k_means_cluster_labels),
              "\t\t",davies_bouldin_score(mPeopleXRestaurantsPcaTransform, k_means_cluster_labels))
        
    return

print("=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=")    
print("I have decided to use random weights and scores for the people and restaurant matrix")
print("As this would be the case in real world and will have an element of surprise")
print("")    

p_names  = ['Ross', 'Rachel', 'Joey', 'Monica', 'Phoebe','Chandler','Jerry', 'George', 'Kramer', 'Elaine']
p_cats = ['Willingness to travel','Desire for new experience', 'Cost', 'Choice of Menu',"Service", 'Environment']

people = make_dict(p_names, p_cats, "F")

M_people = convert_list_to_matrix(people)
print ("=-=-=-=-=-=-=-=-= People Names =-=-=-=-=-=-=")
print(p_names)
print("")
print("=-=-=-=-=-=-=-=-= Category =-=-=-=-=-=-==-=-=")
print(p_cats)
print("")

print("Transform the user data into a matrix(M_people). Keep track of column and row ids")
print ("=-=-=-=-=-=-=-=- People Matrix =-=-=-=-=-=-=")
print (M_people)
print("Each row represents a person and each column is a category in the above order")
print("p_names and p_cats are the row and column headers ")
print('')

r_names  = ['Flacos', 'PF Changs', 'Madeo', 'Souplantation', 'TGI Friday', 'The Stand','Lamandier','Amelie','Fiesta','Chilis']
r_cats = ['Distance', 'Novelty', 'Cost', "Food-Beverage Options", "Staff", "Ambience"]

restaurants = make_dict(r_names,r_cats, "I")

M_restaurants = convert_list_to_matrix(restaurants)
print("=-=-=-=-=-=-= Restaurants Names =-=-=-=-=-=-=-=-=")
print(r_names)
print("")
print("=-=-=-=-=-=-=-=-= Category =-=-=-=-=-=-=-=-=-=-=-=")
print(r_cats)

print("")
print("Transform the restaurant data into a matrix(M_resturants) use the same column index.")
print("=-=-=-=-=-=-=-=- Restaurants Matrix =-=-=-=-=-=-=")
print(M_restaurants)
print("Each row represents a restaurant and each column is a category in the above order")
print("r_names and r_cats are the row and column headers ")
print('')

print("The most important idea in this project is the idea of a linear combination.")
print("Informally describe what a linear combination is and how it will relate to our restaurant matrix.")
print("")
print("Linear Combination is the process of simplifying two algebraic equation so that one variable is eliminated.")
print("In this People vs. Restaurant metrics the weights and ratings are simplified for arriving at the rank")

print("")

M_people_X_restaurants, rankMatrix, M_usr_x_rest_rank = DataProcessing(M_people, M_restaurants, r_names)

print("Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people")
print("=-=-=-=-=-=-= Restaurants X People =-=-=-=-=-=-=-=-=")

print(M_people_X_restaurants)

print("")
print("What does the a_ij matrix represent?")
print("Each Rows represents Restaurants and Column represents the People")
print("")

print("Sum all columns in M_usr_x_rest to get optimal restaurant for all users, What do the entries represent?")
print("Each entry represents overall score of each restaurants by all users, which is the raw score out of 100")
print("")
print(r_names)
print(np.sum(M_people_X_restaurants, axis=1))
print("")

print("Choose a person and compute(using a linear combination) the top restaurant for them, What does each entry in the resulting vector represent.")
print("Below resulting vector represents the favorite restaurant for the person")
print("=-=-=-=-=-= Most Favorite Restaurant of =-=-=-=-=-=")

M_restaurant_max = np.argmax(M_people_X_restaurants, axis=1);
M_people_max = np.argmax(M_people_X_restaurants, axis=0);

for i in range(len(M_people_max)):
    print (p_names[i], "is", r_names[M_people_max[i]])

print("")    

print("Which restaurant got max ratings? and by whom?")

for i in range(len(M_restaurant_max)):
    print (r_names[i], "got high score from ", p_names[M_restaurant_max[i]])

print("")    
print("Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.") 
print("Do the same as above to generate the optimal restaurant choice")
print("")
print("=-=-=-=-=-= Ranks of Restaurants by all People =-=-=-=-=-=-=-=")
print("1: Least & 10: Most Favorite Restaurant")
print("")
print(M_usr_x_rest_rank)
print("")

print("Why is there a difference between the two?, What problem arrives?")
print("A restaurant might get higher score from a person which doesn't mean that is their favorite")
print("Low score across one restaurants creates bias makes the rank out of sequence")
print("")
print("What does represent in the real world?")
print("It is difficult to control individual choice as it is based on the individual experience.")
print("With less number of examples and outliers may makes the recommendation less reliable")

print("")
# Code reference:
# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

fig, ax = plt.subplots(figsize=(8, 8))
plt.imshow(M_people_X_restaurants)
ax.set_xticks(np.arange(len(r_names)))
ax.set_yticks(np.arange(len(p_names)))

ax.set_xticklabels(r_names)
ax.set_yticklabels(p_names)

plt.setp(ax.get_xticklabels(), rotation=35, ha="right",
         rotation_mode="anchor")

for i in range(len(p_names)):
    for j in range(len(r_names)):
        text = ax.text(j, i, round(M_people_X_restaurants[i, j],2),
                       ha="center", va="center", color="w")

ax.set_title('People Vs. Restaurants Scores') 
fig.tight_layout()
plt.show()
plt.close()

print("How should you preprocess your data to remove this problem.")
print("Could remove the outlier (scored very low) to improves the recommendation")
print("Overall Score could be used to decide on while making decisions")
print("Check if there are any issue with these scores for errors.")

print("")
print("Find user profiles that are problematic, explain why?")
print("Heat map created on the matrix would help identify the person who made those choices for further action/decision")
print("Mostly those who score low across creates skewness.")

# Check for optimal clusters
chkMaxClusters(M_people_X_restaurants)

print("Given the size of the data and random weights and score, 2 (or) 3 clusters are optimal choice") 
print("Decided to try two & three clusters for PCA + Kmeans analysis")
print("")


# KNN with PCA.
range_of_components = [2,3]
for n_components in range_of_components:

    pca = PCA(n_components=n_components)   
    subtitle = ("KMeans with PCA - %i" %n_components)
    mPeopleXRestaurantsPcaTransform = pca.fit_transform(M_people_X_restaurants)  
    knn2to4(mPeopleXRestaurantsPcaTransform,subtitle)

print("")
print("Created Dentogram to visualize the similarity and dissimilarity between peoples / restaurants")
print("")
title = "Restaurants Dentogram"
dentogram(M_restaurants, r_names, title)

title = "People Dentogram"
dentogram(M_people, p_names, title)

print("Since the ground truth scores are not available with People X  Restaurant Matrix")
print("Decided to use Davies Bouldin Score & Calinski Harabaz Score to check the purity of the cluster")

checkClusterPurity(M_people_X_restaurants)

print("")                                        

print("These two scores also confirms the 2 (or) 3 clusters are better and good depending upon the data.")
print("Refer to the guidelines on the scores")
print("")
print ("Think of two metrics to compute the dissatisfaction with the group.")
print("Created the low score matrices to identify the least favorite")
print("")

M_restaurant_min = np.argmin(M_people_X_restaurants, axis=1);
M_people_min = np.argmin(M_people_X_restaurants, axis=0);

print("")
print("=-=-=-=-=-= Least Favorite Restaurant of =-=-=-=-=-=")
for i in range(len(M_people_min)):
    print (p_names[i], "is", r_names[M_people_min[i]])

print("")
for i in range(len(M_restaurant_min)):
    print (r_names[i], "got low score from ", p_names[M_restaurant_min[i]])

print("")
print("Should you split in two groups today?")
print("Yes, On multiple runs there were some resulted in two way split since the weights and scores are random") 
print("")
print("If there had to be one group, one option is to try assigning higher weightage to distance") 
print("In real world this could be a deciding factor")
print("")

print("Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?")
print("!! Awesome, make the cost weight from people matrix to zero and/or restaurant matrix score to one and recalculate the rank.")
print("")

newM_people = np.copy(M_people)
newM_people[:, 2] = 0

newM_restaurants = np.copy(M_restaurants)
newM_restaurants[:, 2] = 1

newM_people_X_restaurants, newrankMatrix, newM_usr_x_rest_rank = DataProcessing(newM_people, newM_restaurants, r_names)

print("=-=-=-=-= Restaurants Rank by all People (Boss is paying) =-=-=-=-=-=-=")
print(newM_usr_x_rest_rank)

print("=-=-=-=-= Restaurants Rank by all People (Boss is NOT paying) =-=-=-=-=-=-=")
print(M_usr_x_rest_rank)

print("")
print("As you can clearly see the shift in ranking.")

print("Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.")
print("Can you find their weight matrix?")
print("Yes, Using matrix inverse, we can figure out weight matrix..")
print("Below is the snapshot of the original and inverse weights..")

people = make_dict(p_names, p_cats, "F")
restaurants = make_dict(r_names,r_cats, "I")

M_people_X_restaurants, rankMatrix, M_usr_x_rest_rank = DataProcessing(M_people, M_restaurants, r_names)

M_restaurants_inv = np.linalg.pinv(M_restaurants)
M_people_weights = np.dot(M_restaurants_inv, M_people_X_restaurants)
M_people_weights = np.swapaxes(M_people_weights, 0, 1)

print("People Weight => Original")
print(M_people)
print("")
print("People Weight => using inverse")
print(np.around(M_people_weights,2))
print("")
print("If we do not know the Restaurant Matrix, then we could use np.linalg.lstsq and calculate the least square estimation to arrive at the weights")
sys.stdout = orig_stdout
f.close()

import pdb; pdb.set_trace()