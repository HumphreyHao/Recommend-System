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

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
train_Interactions = readCSV("train_Interactions.csv.gz")

for user,book,r in train_Interactions:
  r = int(r)
  allRatings.append(r)
  userRatings[user].append(r)

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')
  if u in userAverage:
    predictions.write(u + '-' + b + ',' + str(userAverage[u]) + '\n')
  else:
    predictions.write(u + '-' + b + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in train_Interactions:
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

predictions = open("predictions_Read.txt", 'w')
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')
  if b in return1:
    predictions.write(u + '-' + b + ",1\n")
  else:
    predictions.write(u + '-' + b + ",0\n")

predictions.close()

### Category prediction baseline: Just consider some of the most common words from each category

catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}

predictions = open("predictions_Category.txt", 'w')
predictions.write("userID-reviewID,prediction\n")
for l in readGz("test_Category.json.gz"):
  cat = catDict['fantasy_paranormal'] # If there's no evidence, just choose the most common category in the dataset
  words = l['review_text'].lower()
  if 'children' in words:
    cat = catDict['children']
  if 'comic' in words:
    cat = catDict['comics_graphic']
  if 'fantasy' in words:
    cat = catDict['fantasy_paranormal']
  if 'mystery' in words:
    cat = catDict['mystery_thriller_crime']
  if 'young' in words:
    cat = catDict['young_adult']
  predictions.write(l['user_id'] + '-' + l['review_id'] + "," + str(cat) + "\n")

predictions.close()
