import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import ast
import gzip
from urllib.request import urlopen
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')

def kNN(k,u_i):
    total=len(u_i)
    k_neighbor = defaultdict(set)
    for u1 in range(total):
        similarity = [0]*total
        for u2 in range(total):
            similarity[u2] = float(len(u_i[u1] & u_i[u2])) / (len(u_i[u1])+len(u_i[u2]))
        for count in range(k):
            max_similarity_neighbor = similarity.index(max(similarity))
            k_neighbor[u1].add(max_similarity_neighbor)
            similarity[max_similarity_neighbor] = 0
    return k_neighbor

f = gzip.open("train_Interactions.csv.gz", 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split(',')

datatrain = []
datavalid = []
count=0
dataset=[]
for line in f:
    fields = line.strip().split(',')
    d = dict(zip(header, fields))
    dataset.append(d)

random.shuffle(dataset)
lenAll= len(dataset)
lenT =10000
for d in dataset:
    if count <lenT :
        datatrain.append(d)
    else:
        datavalid.append(d)
    count=count+1


users_rate_count = {}
items_rate_count = {}
user_index = []
item_index = []
users_items = defaultdict(set)
#a=0
for datum in datatrain:
    if datum['userID'] not in users_rate_count:
        users_rate_count[datum['userID']] = 1
    else:
        users_rate_count[datum['userID']] += 1
    if datum['bookID'] not in items_rate_count:
        items_rate_count[datum['bookID']] = 1
    else:
        items_rate_count[datum['bookID']] += 1
    if datum['userID'] not in user_index:
        user_index.append(datum['userID'])
    if datum['bookID'] not in item_index:
        item_index.append(datum['bookID'])

for datum in datatrain:
    u_index = user_index.index(datum['userID'])
    i_index = item_index.index(datum['bookID'])
    users_items[u_index].add(i_index)


users_k_neighbor = kNN(1800,users_items)

X={}
Y={}
user_count = {}
for user in range(len(user_index)):
    X[user] = {}
    Y[user] = {}
    weight=len(users_items[user])/10.642260416112382
    prefer_items = {}
    for neighbor in users_k_neighbor[user]:
        for item in users_items[neighbor]:
            if item not in prefer_items.keys():
                prefer_items[item] = 1.0/len(users_items[neighbor])
            else:
                prefer_items[item] += 1.0/len(users_items[neighbor])
    for i in prefer_items:
        X[user][i] = prefer_items[i]/weight
        Y[user][i] = 1 if i in users_items[user] else 0
    
count = 0
for i in range(len(X)):
    if isinstance(X[i], list):
        count += 1
        print (i)
print (count)
print (len(users_k_neighbor[5611]))
'''
for threshould in [1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9,0.89,0.8,0.7,0.6,0.5]:
    predictions = open("predictions_Read.txt", 'w')
    for l in open("pairs_Read.txt",'r'):
        if l.startswith("userID"):
            #header
            predictions.write(l)
            continue
        username,itemname = l.strip().split('-')
        if username in user_index and itemname in item_index:
            u_index = user_index.index(username)
            i_index = item_index.index(itemname)
            if i_index in X[u_index].keys():
                if Y[u_index][i_index] == 1:
                    prediction = 1
                else:
                    sorted(X[u_index].items(), key=lambda item:item[1] , reverse=True)
                    if X[u_index].keys().index(i_index) < len(X[u_index])*threshould:
                        prediction = 1
                    else:
                        prediction = 0
            else:
                prediction = 0
        else:
            prediction = 0

        if prediction == 1:
            predictions.write(username + '-' + itemname + ",1\n")
        else:
            predictions.write(username + '-' + itemname + ",0\n")
    predictions.close()
    '''