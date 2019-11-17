#尝试用rating构造新的特征
#先把文件读出来
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
def JaccardB(book1,book2,bookAllUser,return1):
    if book1 not in return1:
        return 0
    s1 = bookAllUser[book1]
    s2 = bookAllUser[book2]
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom
def JaccardU(user1,user2,userReadBook,return2):
    if user1 not in return2:
        return 0
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
    if rating_average != 0:
        return rating_average
    else:
        return aveUser
#把train和valid训练集划分好，本次valid集合直接取消，因为采用的样本点越多，最后的准确率越高
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
lenT =190000
for d in dataset:
    if count <lenT :
        datatrain.append(d)
    else:
        datavalid.append(d)
    count=count+1
#统计train集合中的数据
userReadBook = defaultdict(set)
bookAllUser  = defaultdict(set)
bookRating =defaultdict(set)
userRating =defaultdict(set)
totalRating=0
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    userReadBook[user].add(book)
    bookAllUser[book].add(user)
    bookRating[book].add(r)
    userRating[user].add(r)
    totalRating+=int(r)
users = len(userReadBook)
books = len(bookAllUser)
aveUser=totalRating/users
aveBook=totalRating/books
###计算流行程度

#生成valid集合的相反集合
#得到了当前的所有数据，开始生成新train
i=0
for d in datavalid:
    if i<lenAll-lenT:
        d['rating'] = 1
        dd = dict(zip(header, fields))
        dd['userID'] = d['userID']
        dd['bookID'] = findBook(d['userID'],userReadBook,bookAllUser)
        dd['rating'] = 0
        datavalid.append(dd)
        i=i+1
    else:
        break


bookCount = defaultdict(int)
totalRead = 0
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    bookCount[book] += 1
    totalRead += 1

mostPopular1 = [(bookCount[x], x) for x in bookCount]
mostPopular1.sort()
mostPopular1.reverse()
#给user一样统计popular
userCount = defaultdict(int)
totalRead = 0
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    userCount[user] += 1
    totalRead += 1

mostPopular2 = [(userCount[x], x) for x in userCount]
mostPopular2.sort()
mostPopular2.reverse()

threshold =70
thresholdU =35
for threshold in range(30,70,10):
    return1 = set()
    count = 0
    for ic, i in mostPopular1:
        count += ic
        return1.add(i)
        if count > totalRead*(threshold/100) : 
            break

    return2 = set()
    count = 0
    for ic, i in mostPopular2:
        count += ic
        return2.add(i)
        if count > totalRead*(thresholdU/100) : 
            break
    #设计feature
    ##首先是pop，暂时选择使用01进行分割，threshold选择,调试threshold
    ##其次是这本书对这个user的相似度，计算所有simi的和(最大值失败)
    ##计算人和书的相似度, 成功
    ##增加两个rating作为feature
    #
    ###计算相似程度，可以放在生成部分直接生成

    X=[]
    y=[]
    for d in datavalid:
        user,book,r =d['userID'],d['bookID'],d['rating']
        sum_similarJ=0
        '''
        sum_similarU=0
        for u in bookAllUser[book] :
            sum_similarU += JaccardU(u,user,userReadBook,return2)
        x1 = sum_similarU
        '''
        for b in userReadBook[user] :
            sum_similarJ += JaccardB(b,book,bookAllUser,return1)
        x2 = sum_similarJ
        #rating
        x5=calBookRating(book,bookRating,aveBook)
        x6=calUserRating(user,userRating,aveUser)
        y1 = int(r)>0
        X.append([1]+[x2]+[x5]+[x6])
        y.append(y1)
    model = linear_model.LogisticRegression(C=1, class_weight='balanced',solver='lbfgs')
    model.fit(X, y)

    predictions = model.predict(X)
    TP_ = numpy.logical_and(predictions, y)
    FP_ = numpy.logical_and(predictions, numpy.logical_not(y))
    TN_ = numpy.logical_and(numpy.logical_not(predictions), numpy.logical_not(y))
    FN_ = numpy.logical_and(numpy.logical_not(predictions), y)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    acc =(TP + TN) / (TP + FP + TN + FN)
    print (threshold,thresholdU,"the accuracy is ",acc);

'''
#打开文件正式开始测试
X1=[]
predictions = open("predictions_Read.txt", 'w')
for l in open("pairs_Read.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    user,book = l.strip().split('-')
    #要算出这一行对应的X和y
    sum_similarJ=0
    sum_similarU=0
    for u in bookAllUser[book] :
            sum_similarU += JaccardU(u,user,userReadBook,return2)
            x1 = sum_similarU
    for b in userReadBook[user] :
        sum_similarJ += JaccardB(b,book,bookAllUser,return1)
        x2 = sum_similarJ
    x5=calBookRating(book,bookRating,aveBook)
    x6=calUserRating(user,userRating,aveUser)
    X1.append([1]+[x1]+[x2]+[x5/x6])
pre = model.predict(X1)
pre1=[]
for p in pre:
    if p == False:
        p='0'
    else :
        p='1'
    pre1.append(p)
i =0 
for l in open("pairs_Read.txt"):
    if l.startswith("userID"):
        #header
        #predictions.write(l)
        continue
    user,book = l.strip().split('-')
    predictions.write(user + '-' + book + ","+pre1[i]+"\n")
    i+=1
predictions.close()
'''