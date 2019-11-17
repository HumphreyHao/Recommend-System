#Knn算法
#use the whole train to build a Knn model and save all the item each user visit
##get the train set and valid set to test
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
import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from sklearn import svm
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

def findBook(user,userReadBook,bookAllUser):
    l3 = [x for x in list(bookAllUser) if x not in userReadBook[user]]
    proxy = random.choice(l3)
    return proxy
def JaccardB(book1,book2,bookAllUser):
    s1 = bookAllUser[book1]
    s2 = bookAllUser[book2]
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom
def JaccardU(user1,user2,userReadBook):
    s1 = userReadBook[user1]
    s2 = userReadBook[user2]
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom
def calBookRating(book,bookRating,aveBook):
    if book not in bookRating:
        return aveBook
    rating_average =0
    for d in bookRating[book]:
        rating_average+=int(d)
    rating_average=rating_average/len(bookRating[book])
    return rating_average
def calUserRating(user,userRating,aveUser):
    if user not in userRating:
        return aveUser
    rating_average =0
    for d in userRating[user]:
        rating_average+=int(d)
    rating_average=rating_average/len(userRating[user])
    return rating_average

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
lenT =100
for d in dataset:
    if count <lenT :
        datatrain.append(d)
    else:
        datavalid.append(d)
    count=count+1
#cal the Jac sim
userReadBook = defaultdict(set)
bookAllUser  = defaultdict(set)
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    userReadBook[user].add(book)
    bookAllUser[book].add(user)
#go through the whole set to cal the similarity with each user,and choose the high
#Kth most similar ones
#去重
userKneighbor = defaultdict(set)
user_Kth = defaultdict(dict)
user_hashset=[]
K=100
for userX in userReadBook:
    for userY in userReadBook:
        if userX == userY:
            continue
        else:
            string1 = userX+'_'+userY
            string2 = userY+'_'+userX
            if string1 not in user_hashset and string2 not in user_hashset :
                simXY=JaccardU(userX,userY,userReadBook)
                user_Kth[userX]=[simXY,userY]
                user_Kth[userY]=[simXY,userX]
                user_hashset.append(string1)
                user_hashset.append(string2)
            else:
                continue
for userX in user_Kth:
    mostPopular = [[simXY,user] for simXY,user in user_Kth[userX]]
    mostPopular.sort()
    mostPopular.reverse()
    count =0
    for x_sim,x in mostPopular:
        if count<K:
            userKneighbor[userX].add(x)
            count+=1
        else:
            break
        
#cal the item this cluster has visited and give the score

#for each pair, check its scores and give the judgement with a threshold.