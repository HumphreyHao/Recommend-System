import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict # Dictionaries with default values
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import ast
import gzip
def Jaccard(book1,book2,bookAllUser):
    s1=[]
    for d in bookAllUser[book1]:
        s1.append(d['userID'])
    s2=[]
    for d in bookAllUser[book2]:
        s2.append(d['userID'])
    s1=set(s1)
    s2=set(s2)
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom
#把train和valid训练集划分好
f = gzip.open("train_Interactions.csv.gz", 'rt', encoding="utf8")
header = f.readline()
header = header.strip().split(',')
datatrain = []
datavalid = []
count=0
lenAll = 200000
for line in f:
    fields = line.strip().split(',')
    d = dict(zip(header, fields))
    if count <lenAll*0.95 :
        datatrain.append(d)
    else:
        datavalid.append(d)
    count=count+1

#统计当前看书情况
userReadBook = defaultdict(list)
bookAllUser  = defaultdict(list)

#统计每个看过的书的集合。
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    userReadBook[user].append(d)
    bookAllUser[book].append(d)
#计算平均rating
ratingMean = sum([float(d['rating']) for d in datatrain]) / len(datatrain)

N = len(datatrain)
nUsers = len(userReadBook)
nBooks = len(bookAllUser)
users = list(userReadBook)
books = list(bookAllUser)

alpha = ratingMean

userBiases = defaultdict(float)
bookBiases = defaultdict(float)

def prediction(user, book):
    return alpha + userBiases[user] + bookBiases[book]

def predictRating(user,item):
    ratings = []
    similarities = []
    for d in userReadBook[user]:
        i2 = d['bookID']
        if i2 == item: continue
        ratings.append(d['rating'])
        similarities.append(Jaccard(item,i2,bookAllUser))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings)
    else:
        # User hasn't rated any similar items
        return ratingMean

def unpack(theta):
    global alpha
    global userBiases
    global bookBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    bookBiases = dict(zip(books, theta[1+nUsers:]))

def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['userID'], d['bookID']) for d in datatrain]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in bookBiases:
        cost += lamb*bookBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(datatrain)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dbookBiases = defaultdict(float)
    for d in datatrain:
        u,i = d['userID'], d['bookID']
        pred = prediction(u, i)
        diff = pred - float(d['rating'])
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dbookBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in bookBiases:
        dbookBiases[i] += 2*lamb*bookBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dbookBiases[i] for i in books]
    return numpy.array(dtheta)

def MSE(predictions, labels):
        differences = [(x-y)**2 for x,y in zip(predictions,labels)]
        return sum(differences) / len(differences)

labels = [float(d['rating']) for d in datatrain]
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nBooks),derivative, args = (labels, 2e-5))

#现在已经得到了两个贝塔表，需要解决的问题是，如果这个人或者书没有在表中出现，需要退化成相似度模型
prediction_valid = []
for d in datavalid:
    if d['userID'] in users and d['bookID'] in books:
        prediction_valid.append(prediction(d['userID'],d['bookID']))
    else:
        #退化成相似度模型
        prediction_valid.append(predictRating(d['userID'],d['bookID']))
#求出了预测
labels = [float(d['rating']) for d in datavalid]
print(MSE(prediction_valid,labels))

