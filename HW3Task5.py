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

#把train和valid训练集划分好
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
for d in dataset:
    if count <190000 :
        datatrain.append(d)
    else:
        datavalid.append(d)
    count=count+1


userReadBook = defaultdict(set)
bookAllUser  = defaultdict(set)

#统计每个看过的书的集合。
for d in datatrain:
    user,book,r =d['userID'],d['bookID'],d['rating']
    userReadBook[user].add(book)
    bookAllUser[book].add(user)

#得到了当前的所有数据，开始生成新valid 
i=0
for d in datavalid:
    if i<10000:
        dd = dict(zip(header, fields))
        dd['userID'] = d['userID']
        dd['bookID'] = findBook(d['userID'],userReadBook,bookAllUser)
        dd['rating'] = 0
        datavalid.append(dd)
        i=i+1
    else:
        break

#对每一条记录，找到所有的本user对应的book集合B，对其中的每一个b，计算雅可比
##雅可比计算是比较书对应的user的相似度
#如果说，某一本书和此书的相似度非常高，那么设定本书为read（因为这个user读过类似的书）
#加入popularity相关的判定
bookCount = defaultdict(int)
totalRead = 0
for d in datatrain:
      user,book,r =d['userID'],d['bookID'],d['rating']
      bookCount[book] += 1
      totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalRead*0.64: break

#使用对应的算法来完成对文件读取的每一行的预测
predictions = open("predictions_Read.txt", 'w')
thresholdJ=0.01
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  user,book = l.strip().split('-')
  flag ='0'
  sum_similarJ=0
  flagP =0
  flagB =0
  flagU =0
  if book in return1:
      flagP =1
  for b in userReadBook[user] :
        sum_similarJ += JaccardB(b,book,bookAllUser)
        if sum_similarJ > thresholdJ:
                flagB=1
                break
  for u in bookAllUser[book] :
        sum_similarJ += JaccardU(u,user,userReadBook)
        if sum_similarJ > thresholdJ:
                flagU=1
                break   
  if flagP ==1 and flagB ==1 and flagU ==1 :
        flag ='1'
  predictions.write(user + '-' + book + ","+flag+"\n")
predictions.close()