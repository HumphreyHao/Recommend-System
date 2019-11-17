import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict 
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import ast
import gzip

def readGz(path):
      for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

def findBook(user,userReadBook,bookAllUser):
    l3 = [x for x in list(bookAllUser) if x not in userReadBook[user]]
    proxy = random.choice(l3)
    return proxy
#先根据训练集去找到最流行的书，训练成一个模型
#测试集本身是和训练集的后半部分组成，原来就有的记录标记为1，
#其中的每一个userid插入一条任意的没看过的书，记录标记为0
##如何实现，把valid集合统计出一个这个user看过的书的集合，和这个user没看过的书的集合，
##对每一条语句生成一个随机的（可以重复）反语句
#判断当前的记录是否在模型中出现，如果出现则标记为1，没出现则标记为0，统计准确率


#把train和valid训练集划分好
f = gzip.open("train_Interactions.csv.gz", 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split(',')

datatrain = []
datavalid = []
count=0
for line in f:
    fields = line.strip().split(',')
    d = dict(zip(header, fields))
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

#生成好，开始用train训练模型
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
  if count > totalRead/2: break

#训练完成，return1即为所求
count =0
prediction =[]
for d in datavalid:
    user,book,r =d['userID'],d['bookID'],d['rating']
    if book in return1  :
        prediction.append(1) 
    else:
        prediction.append(0)
    count=count+1
#得到了prediction值，计算准确度
count =0
Tcount=0
for d in datavalid:
    if prediction[count] >0 and int(d['rating'])>0:
        Tcount+=1
    if prediction[count] ==0 and int(d['rating'])==0:
        Tcount+=1
    count+=1
accuracy = Tcount/len(prediction)
print(accuracy)