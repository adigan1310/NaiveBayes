import os
import math
import re
import random
from stop_words import get_stop_words

#authors: Adithya Ganapathy and Nisshanthni Divakaran
#UTD ID: axg172330 and nxd171330

naivebayesnet = {}

#Function to read all subfolders and files in the directories and
#creates the vocab list, doc number and the classes.
def train(path):
    cls = {}
    vocab = []
    td = os.listdir(path)
    doc = 0
    stop_words = get_stop_words('english')
    doclist1 = []
    for row in td:
        doclist1.append(row)
    doclist = random.sample(doclist1,5)
    for row in doclist:
        subpath = path + '/' + row
        td = os.listdir(subpath)
        cls[row] = {}
        cls[row]['doc'] = doc
        wordlist = {}
        cwc = 0
        for file in td:
            filepath = subpath + '/' + file
            data = open(filepath, 'r').read().split('Lines')
            if len(data) > 1:
                data = data[1]
            else:
                data = data[0]
            index = data.index('\n')
            data = data[index:]
            tokens = re.split(r'\W', data)
            tokens = [token.lower() for token in tokens if (len(token) > 1)]
            filtereddata = [x for x in tokens if x not in stop_words]
            vocab.extend(filtereddata)
            for token in filtereddata:
                if token in wordlist:
                    wordlist[token] = wordlist[token] + 1
                    cwc = cwc + 1
                else:
                    wordlist[token] = 2             #Laplace Smoothing
                    cwc = cwc + 2
            doc = doc + 1
        cls[row]['doc'] = doc - cls[row]['doc']
        cls[row]['word'] = wordlist
        cls[row]['wordcount'] = cwc
    print("Chosen List:")
    print(doclist)
    vocab = list(set(vocab))
    return vocab,doc,cls

#Function that calculates the conditional prob,prior and also
#calls the train function which returns vocab list,doc numbers
#and classes.
def trainnb(path):
    global naivebayesnet
    vocab,totaldocs,cls = train(path)
    prior = {}
    likelihood = {}
    for rows in cls:
        prior[rows] = cls[rows]['doc']/totaldocs
        denom = cls[rows]['wordcount']
        word = cls[rows]['word']
        for words in vocab:
            if words in word:
                num = word[words]
            else:
                num = 1
            if words not in likelihood:
                likelihood[words] = {}
            likelihood[words][rows] = num/denom
    naivebayesnet['vocab'] = vocab
    naivebayesnet['prior'] = prior
    naivebayesnet['likelihood'] = likelihood
    naivebayesnet['class'] = cls.keys()

#Function that calculates the score of a class and
#predicts the output class based on the max(score)
#among all classes.
def testnbdoc(cls, prior, likelihood, word):
    score = -99999999
    line = ""
    for row in cls:
        comp = math.log(prior[row])
        for w in word:
            comp = comp + math.log(likelihood[w][row])
        if comp > score:
            score = comp
            line = row
    return line

#Function that reads the files
#and gets the tokens from the files present in the folders.
#the intersection of vocab list and obtained list is returned.
def test(vocab, subpath):
    data = open(subpath, 'r').read().split('Lines')
    stop_words = get_stop_words('english')
    if len(data) > 1:
        data = data[1]
    else:
        data = data[0]
    index = data.index('\n')
    data = data[index:]
    tokens = re.split(r'\W', data)
    tokens = [token.lower() for token in tokens if len(token) > 1]
    filtereddata = [x for x in tokens if x not in stop_words]
    tokens = list(set(vocab).intersection(set(filtereddata)))
    return tokens

#Function that reads the sub folders and its path and calculates
#the accuracy of the Naive bayes classifier and prints it.
def testnb(testingPath):
    global naivebayesnet
    cls = naivebayesnet['class']
    prior = naivebayesnet['prior']
    likelihood = naivebayesnet['likelihood']
    vocab = naivebayesnet['vocab']
    correct = 0
    wrong = 0
    for row in cls:
        path = testingPath + '/' + row
        dir = os.listdir(path)
        for file in dir:
            word = test(vocab , path + '/' + file)
            predictoutput = testnbdoc(cls , prior, likelihood , word)
            if predictoutput == row:
                correct = correct + 1
            else:
                wrong = wrong + 1
    print('Accuracy of the Multinomial Naive Bayes Classifier is: ', str((correct/(correct+wrong))*100))

#main function
trainingpath, testingpath = input().split()
trainnb(trainingpath)
testnb(testingpath)