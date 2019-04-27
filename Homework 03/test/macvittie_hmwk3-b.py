import numpy as np
from collections import OrderedDict
import json
import matplotlib.pyplot as plt

import sys
orig_stdout = sys.stdout
f = open('MacVittie_Hmwk3_Output.txt', 'w')
sys.stdout = f

print("MACVITTIE - HOMEWORK 3")

np.random.seed(0xdedede) # just in case, yo

# values of columns for each person should total to 1
people = {'Claire': {'Willingness to Travel': 0.24, 'Desire for New Experience': 0.43, 'Cost': 0.06, 'Sensitivity to Ratings': 0.06, 'Pretension': 0.21}, 
			'Angel': {'Willingness to Travel': 0.02, 'Desire for New Experience': 0.34, 'Cost': 0.25, 'Sensitivity to Ratings': 0.35, 'Pretension': 0.04}, 
			'Val': {'Willingness to Travel': 0.47, 'Desire for New Experience': 0.07, 'Cost': 0.01, 'Sensitivity to Ratings': 0.29, 'Pretension': 0.16}, 
			'Beatrix': {'Willingness to Travel': 0.17, 'Desire for New Experience': 0.12, 'Cost': 0.39, 'Sensitivity to Ratings': 0.05, 'Pretension': 0.27}, 
			"Y'shtola": {'Willingness to Travel': 0.16, 'Desire for New Experience': 0.06, 'Cost': 0.48, 'Sensitivity to Ratings': 0.17, 'Pretension': 0.13}, 
			'Roxy': {'Willingness to Travel': 0.24, 'Desire for New Experience': 0.48, 'Cost': 0.11, 'Sensitivity to Ratings': 0.1, 'Pretension': 0.07}, 
			'Moth': {'Willingness to Travel': 0.14, 'Desire for New Experience': 0.17, 'Cost': 0.17, 'Sensitivity to Ratings': 0.03, 'Pretension': 0.49}, 
			'Troi': {'Willingness to Travel': 0.3, 'Desire for New Experience': 0.21, 'Cost': 0.06, 'Sensitivity to Ratings': 0.07, 'Pretension': 0.36}, 
			'Olivia': {'Willingness to Travel': 0.28, 'Desire for New Experience': 0.23, 'Cost': 0.31, 'Sensitivity to Ratings': 0.03, 'Pretension': 0.15}, 
			'Lilith': {'Willingness to Travel': 0.2, 'Desire for New Experience': 0.09, 'Cost': 0.13, 'Sensitivity to Ratings': 0.22, 'Pretension': 0.36}
		}
		
# each feature on a 1-10 scale
restaurants = {'The Bottle': {'Distance': 4, 'Novelty': 7, 'Cost': 9, 'Rating': 4, 'Atmosphere': 7}, 
				"Humphrey's": {'Distance': 3, 'Novelty': 2, 'Cost': 5, 'Rating': 9, 'Atmosphere': 6}, 
				"Maggie Meyer's": {'Distance': 8, 'Novelty': 6, 'Cost': 1, 'Rating': 8, 'Atmosphere': 10}, 
				"Keegan's": {'Distance': 3, 'Novelty': 3, 'Cost': 1, 'Rating': 8, 'Atmosphere': 4}, 
				'Grille 29': {'Distance': 10, 'Novelty': 8, 'Cost': 6, 'Rating': 6, 'Atmosphere': 7}, 
				'Caffe Espresso': {'Distance': 3, 'Novelty': 4, 'Cost': 4, 'Rating': 10, 'Atmosphere': 7}, 
				"Anduzzi's": {'Distance': 7, 'Novelty': 7, 'Cost': 5, 'Rating': 2, 'Atmosphere': 5},
				'Cotton Row': {'Distance': 1, 'Novelty': 2, 'Cost': 3, 'Rating': 5, 'Atmosphere': 8}, 
				'Republic Chophouse': {'Distance': 9, 'Novelty': 4, 'Cost': 3, 'Rating': 8, 'Atmosphere': 2}
		}

lst_to_mtx = []
people_rows = []
people_cols = [ 'Willingness to Travel', 'Desire for New Experience', 'Cost', 'Sensitivity to Ratings', 'Pretension']

for person, descrip in people.items():
	people_rows.append(person)
	lst_to_mtx.append([descrip[cat] for cat in people_cols])
	
people_mtx = np.matrix(lst_to_mtx)

lst_mtx_r = []
rest_rows = []
rest_cols = [ 'Distance', 'Novelty', 'Cost', 'Rating', 'Atmosphere' ]

for rest, desc in restaurants.items():
	rest_rows.append(rest)
	lst_mtx_r.append([desc[cat] for cat in rest_cols])
	
rest_mtx = np.matrix(lst_mtx_r)

# q 1
print("\n> Q) The most imporant idea in this project is the idea of a linear combination.\n> Informally describe what a linear combination is and how it will relate to our resturant matrix.")
print("\n\tLinear combination takes two equations and simplifies, removing one variable.\n\tHere, we want to take in the preferences of the people and attributes of the restaurants, and arrive at a rank.\n")

# q 2
print("\n> Q) Choose a person and compute (using a linear combination) the top restaurant for them. What does each entry in the resulting vector represent?")

pick = 'Beatrix'

person = people[pick]
p = np.matrix([person[col] for col in people_cols])
prefs = np.dot(p, rest_mtx.T).tolist()[0]

out = { k:v for k,v in zip(rest_rows, prefs) }

for _ in out.items():
	print("\t", _)
	
best_val = 0.0
best_rest = None

for r, s in out.items():
	if s > best_val:
		best_val = s
		best_rest = r
		
print("\n\tThe best restaurant for {} is {}. The values in the vector above represent her preference for a particular restaurant as a combination of her preferences and the restaurant's qualities.".format(pick, best_rest))

# q 3
print("\n> Q) Choose a person and compute(using a linear combination) the top restaurant for them. What does each entry in the resulting vector represent.")

pick = 'Troi'
person = people[pick]
p = np.matrix([person[col] for col in people_cols])
prefs = np.dot(p, rest_mtx.T).tolist()[0]

out = { k:v for k,v in zip(rest_rows, prefs) }

for _ in out.items():
	print("\t", _)
	
print("\n\tThese values represent {}'s level of preference for a given restaurant, based upon their preferences and the restaurant values of the locale in question.".format(pick))

# q 4
print("\n> Q) Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people. What does the a_ij matrix represent?")

M_usr_x_rest = np.dot(people_mtx, rest_mtx.T)

print("\n", M_usr_x_rest)

print("\n\tIn this case, each row represents a person, and each column, a restaurant. So the 'i' value is the row, the 'j' the column; thus 'i' represents the person, while 'j' the restaurant.")

# q 5
print("\n> Q) Sum all columns in M_usr_x_rest to get optimal restaurant for all users. What do the entries represent?")

foo = np.sum(M_usr_x_rest, axis=1)

print(foo)

print("\n\tEach entry represents the sum of each rating from each person, based upon their preferences and the restaurant's qualities, resulting in a total score ranging from 0 to 100. In this case, the higher the score, the better: here it would appear that the second restaurant on the list, 'Humphrey's', should win.")

# q 6
print("\n> Q) Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank. Do the same as above to generate the optimal resturant choice.")

def vals_to_ranks(lst):
	n_vals = []
	preferences = {v:r for r,v in enumerate(sorted(lst))}
	
	for val in lst:
		n_vals.append(preferences[val])
		
	return n_vals

M_usr_x_rest_rank = np.matrix([vals_to_ranks(M_usr_x_rest[row].tolist()[0]) for row in range(M_usr_x_rest.shape[0])])

print(M_usr_x_rest_rank)

scores = [M_usr_x_rest_rank[:,n].sum() for n in range(M_usr_x_rest_rank.shape[1])]
ranks = vals_to_ranks(scores)

out = zip(rest_rows, ranks)

for rsnt, score in out:
	print("\t{}: [{}]".format(rsnt, score)) # adding 1 because it starts at 0

# q ???
print("\n Q) Why is there a difference between the two? What problem arrives? What does represent in the real world?")

print("\n\tWhat's going on here is that in the first matrix, we are using weighted values, while in the second, restaurants are being strictly ordered. For instance, someone might rate two restaurants a 2.12, then a 7.18, with all others being below the low or above the high value - but if ranked, those two values would be translated into rankings immediately nearby. Using strict ranking removes 'strong' opinions, and instead only cares about ordering, rather than a strength of preference.")

# q ???
print("\n Q) How should you preprocess your data to remove this problem.")

print("\n\tI'm not really sure it's a problem, actually. This is just two different ways of looking at the data. In one model, you are taking strength of preference into account; in the other, you're only looking for a ranking, regardless of how strongly a person feels about certain aspects of a restaurant. It may be more useful to take the strong preferences into account, in terms of social cohesion. That said, it is possible that using the first system (using preferences as weights, rather than strict ordering) may be problematic as it would allow an individual with overwhelmingly strong preferences to dominate the model - but again, in real world social situations, if no one else has any particularly strong preferences, that might be acceptable.")

# q ???
print("\n Q) Find user profiles that are problematic, explain why.")

fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(M_usr_x_rest)
ax.set_yticks(np.arange(len(people)))
ax.set_xticks(np.arange(len(restaurants)))

ax.set_yticklabels(people)
ax.set_xticklabels(restaurants)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(10):
	for j in range(9):
		text = ax.text(j, i, round(M_usr_x_rest[i, j],2),
			ha="center", va="center", color="w")

ax.set_title('People Vs. Restaurants Scores') 
fig.tight_layout()
fig.savefig('heatmap.png')
plt.close()

print("\n\tExamining the a heatmap (see 'heatmap.png'), it doesn't look like there are any particularly problematic profiles. If anything it would appear that there may be some issues with restaurants: Keegan's seems universally loathed, while Grille 29 is widely liked.")

print("\n> Q) Think of two metrics to compute the disatistifaction with the group.")

# what we will do is look at a person's least favorite places.
less_than_4 = M_usr_x_rest_rank < 4
scores = [less_than_4[:,n].sum() for n in range(M_usr_x_rest_rank.shape[1])]

out = zip(rest_rows, scores)

for rsnt, score in out:
	print("\t{}: [{}]".format(rsnt, score))
	
print("\n\tThis vector indicates how often a restaurant appears in a person's 'bottom 3' picks: we could call this a 'dissatisfaction score', indicating the level of malcontent with the selection. Here we can see that Keegan's rates a 10, while Grille 29 rates a 0; other restaurants fall within that range. Given how our preferences and restaurant feature ratings fell out, it turns out that the most popular of the restaurants will leave no one unhappy. I wish I could say I planned that, because it'd be funnier that way, but I did not.\n\n\tSo for this particular group, it doesn't make sense to split into two groups: everyone is perfectly fine with Grille 29.")

# another q
print("\n> Q) Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?")

# for this one, we're now going to ignore price.

rests = rest_mtx[:,[0,1,3,4]]
ppl = people_mtx[:,[0,2,3,4]]

M_usr_x_rst_free = np.dot(ppl, rests.T)

scores = [M_usr_x_rst_free[:,n].sum() for n in range(M_usr_x_rst_free.shape[1])]
ranks = vals_to_ranks(scores)

out = zip(rest_rows, ranks)

for rsnt, rank in out:
	print("\t{}: [{}]".format(rsnt, rank))
	
print("\n\tIf cost is no object, then Maggie Meyer's overcomes Grille 29; in addition, there is some shuffling around of the other restaurants, but not much.")

# last q
print("\n> Q) Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants. Can you find their weight matrix?")

print("\n\tNo, because that information has been lost. For instance, a person who rates everything low and a person who rates everything high could very well have the same rank ordering, but their actual preferences are significantly different. There's no way to reconsruct the weight matrix from just the rankings.")