#!/usr/bin/python
import numpy as np
import json
import re
import os
import matplotlib.pyplot as plt
from collections import Counter
from DeathtoGridSearch import GridSearch

# output all our answers to a file
import sys
orig_stdout = sys.stdout
f = open('MacVittie_Hmwk2_Output.txt', 'w')
sys.stdout = f

print("MACVITTIE - HOMEWORK 2\n")

# use regex to read data
filename = "claim.sample.csv"
raw_data = [re.sub("[^a-zA-Z0-9.,]", "", i).split(',') for i in open(filename).read().strip().split("\n")]

# columns and data
columns = raw_data[0]
data = raw_data[1:]

# gimme a mapping for column names to numbers
colmap = {k:v for v,k in enumerate(columns)}

# j-code data
j_idx = []
jdf = []
m_cols = []

### question 1
print("1: A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').")

# 1.a
j_codes = 0

for rownum, row in enumerate(data):
	try:
		if row[colmap['Procedure.Code']][0] == 'J':
			j_codes += 1
			j_idx.append(rownum)
			jdf.append(row)
	except IndexError:
		continue
			
print("\n\ta) Find the number of claim lines that have J-codes.")
print("\t\t", j_codes)

# 1.b
ttl_pay = 0

for row in jdf:
	ttl_pay += float(row[colmap["Provider.Payment.Amount"]])
	
print("\n\tb) How much was paid for J-codes to providers for 'in network' claims?")
print("\t\t${:.2f}".format(ttl_pay))

# 1.c
c_map = {}

for row in jdf:
	try:
		c_map[row[colmap["Procedure.Code"]]] += float(row[colmap["Provider.Payment.Amount"]])
	except KeyError:
		c_map[row[colmap["Procedure.Code"]]] = float(row[colmap["Provider.Payment.Amount"]])
		
code_d = {v:k for k,v in c_map.items()}

print("\n\tc) What are the top five J-codes based on the payment to providers?")
for v in sorted(code_d.keys(), reverse = True)[:5]:
	print("\t\t" + code_d[v] + ": {}".format(v))

### question 2
print('\n\n2: For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.')

# 2.a
unpaid_claims = Counter()
paid_claims = Counter()

for row in jdf:
	if int(float(row[colmap["Provider.Payment.Amount"]])) == 0:
		unpaid_claims[row[colmap["Provider.ID"]]] += 1
	else:
		paid_claims[row[colmap["Provider.ID"]]] += 1
		
x = [paid_claims[k] for k in paid_claims.keys()]
y = [unpaid_claims[k] for k in paid_claims.keys()]

fig, ax = plt.subplots(1, 1, figsize = (20, 20))

ax.scatter(x, y)
ax.set_title("Paid Claims v. Unpaid Claims by Provider")
ax.set_xlabel("Paid Claims")
ax.set_ylabel("Unpaid Claims")
ax.set_xlim(-5, 2000)
ax.set_ylim(-5, 14000)

fig.savefig('macv_q_2a.png')

print('\n\ta) Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.')
print("\t\tChart can be found as 'macv_q_2a.png'.")

# 2.b
print('\n\tb) What insights can you suggest from the graph?')
	
print("\t\tLooks like there are a lot more unpaid claims than paid claims. I'll be honest, I'm not really sure how to interpret that, as I'm not in the medical or insurance industries. These values seem to be all over the place, but there does seem to be at least some correlation between number of paid and number of unpaid claims.")

# 2.c
print("\n\tc) Based on the graph, is the behavior of any of the providers concerning? Explain.")
print("\t\tAgain, that's hard to say without some domain knowledge. I would presume that unpaid claims are bad, but then I'm not sure if a claim is unpaid because the firm in question disagrees that they need to cover it, or perhaps there is some investigation into insurance fraud. If there's not anything unusual going on, then I would presume that the firms with large numbers of unpaid claims are probably doing poorly.")

### question 3
print('\n\n3: Consider all claim lines with a J-code.')

# 3.a
unpaid_cnt = 0
paid_cnt = 0

for row in jdf:
	if int(float(row[colmap["Provider.Payment.Amount"]])) == 0:
		unpaid_cnt += 1
	else:
		paid_cnt += 1
		
unpaid_ratio = unpaid_cnt / (unpaid_cnt + paid_cnt) * 100

print("\n\ta) What percentage of J-code claim lines were unpaid?")
print("\t\t{:.2f}%".format(unpaid_ratio))

# 3.b
# build a model? sounds like the time to bring in code from first assignment

model_cols = []

numeric_cols = ['Subscriber.Payment.Amount', 'Claim.Charge.Amount']

# we have some categoricals, which will need to be one-hot encoded
encoded_cols = { "Provider.ID" : {}, "Line.Of.Business.ID" : {}, "Service.Code": {}, "In.Out.Of.Network" : {}, "Network.ID": {}, "Agreement.ID" : {}, "Price.Index": {}, "Claim.Type": {}, "Procedure.Code": {}, "Revenue.Code": {} }

for rownum, row in enumerate(jdf):
	for col in encoded_cols.keys():
		try:
			encoded_cols[col][rownum].add(row[colmap[col]])
		except KeyError:
			encoded_cols[col][row[colmap[col]]] = {rownum}
			
for colname, encdict in encoded_cols.items():
	for enc in encdict.keys():
		model_cols.append("{}_{}".format(colname, enc))
		
#categoricals encoded, let's continue
model_df = []

for i, r in enumerate(jdf):
	i_r = []
	
	# throw values into the hot-encoded cols
	for colname in model_cols:
		col, val = colname.split("_")
		if i in encoded_cols[col][val]:
			i_r.append(1)
		else:
			i_r.append(0)
	
	for n in numeric_cols:
		i_r.append(float(r[colmap[n]]))
		
	# change out the payment to a binary for classification purposes
	
	if float(r[colmap["Provider.Payment.Amount"]]) > 0.0:
		i_r.append(0.0)
	else:
		i_r.append(1.0)
		
	model_df.append(i_r)

for n in numeric_cols:
	model_cols.append(n)
	
# turn it into a numpy array	
np_df = np.array(model_df)

#print(np_df)
		
#print(np_df[:,-1])

gs = GridSearch(np_df[:,:-1], np_df[:, -1])
gs.optimize_all_models()

print("\n\tb) Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.")
print("\t\tFor starters, we needed to one-hot encode our categoricals. I then used the grid search code from assignment 1 to run the data through a variety of models.")

# 3.c

print("\n\tc) How accurate is your model at predicting unpaid claims?")
print("\t\tDecision trees were generally trash, with an AUC of at most .6, which is pretty bad.")

print("\t\tLogistic regressions struggled to get over a .5, so those are clearly not the best use here.")

print("\t\tRandom forests were simlarly strapped to get results, hitting an AUC of .54 at best.")

print("\t\tXGBoost, however, performed pretty well, with an AUC of .7 using 19 estimators and a max depth of 19. It got similar results with a max depth of 17, though, so I think that's probably as good as it's going to get. Looking at smaller numbers of estimaters, 16 estimators with depth 19 got pretty similar results, so I think that's as good as it's going to get.")

# 3.d

print("\n\td) What data attributes are predominately influencing the rate of non-payment?")
		
print("\t\tSince the best results were from an XGBoost, we can't look at what it was using to decide, so it's hard to say.")

print("\t\tApparently there exists some way to get feature importance out of XGBoost, but I wasn't able to figure it out.")