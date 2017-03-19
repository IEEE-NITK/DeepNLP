import random
import sys

import nltk
import itertools
from collections import defaultdict
import re 

import numpy as np

import pickle

## SETTING UP FLA

MAX_INPUT_LENGTH = 20
#VOCAB_SIZE = 10000

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def filter_data(sequences):
    filtered_q = []
    raw_data_len = len(sequences)

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen <= MAX_INPUT_LENGTH :
            if alen <= MAX_INPUT_LENGTH:
                filtered_q.append(sequences[i])
                filtered_q.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q

## GETTING AND CLEANING DATA

book_path = 'gottext.txt'
read_data = ""

print('Preparing data...')
with open(book_path) as book:
  read_data = book.read()

data = read_data.split('\n')

data = [line.lower() for line in data]
data = [ filter_line(line, EN_WHITELIST) for line in data]

VOCAB_SIZE = len(set(read_data.split())) #Count of unique words

unk = '<unknown>'
go ='<go>'
pad ='<pad>'
eos ='<eos>'

tokenized_sentences=[]
tokenized_sentences = [ wordlist.split(' ') for wordlist in data ]
    

freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
# get vocabulary of 'vocab_size' most used words
vocabulary = freq_dist.most_common(VOCAB_SIZE) #REDUCE/INCREASE THIS SIZE AND CHECK PERFORMANCE
# index2word
index2word = [pad, unk, go, eos] + [ x[0] for x in vocabulary ]
# word2index
word2index = dict([(w,i) for i,w in enumerate(index2word)] )

print('Vocabulary created.')
print("Created vocabulary of " + str(len(word2index)) + " words.")




quotes = re.findall('“([^”]*)”', read_data)
# Use all quotes
quotes = [quote.lower() for quote in quotes]

print("\nSample lines: ")
print(quotes[20:25])

quotes = [ filter_line(line, EN_WHITELIST) for line in quotes]

print("\nLines after processing: ")
print(quotes[20:25])
quotes = filter_data(quotes)

print("\nFiltered lines: ")
print(quotes[20:25])


def sen2enco(sentence):
  if 'gim' in sentence:
    print(sentence)
  return [word2index[word] for word in sentence.split()]  

def getID(word, create=True):
  word = word.lower()
  wid = vocab["word2id"].get(word, -1)
  if wid == -1:
    if create:
      wid = len(vocab["word2id"])
      vocab["word2id"][word] = wid
    else:
      wid = vocab["word2id"].get("<unknown>")
  return wid

print('Creating samples...')
onehot_quotes = [sen2enco(quote) for quote in quotes]

samples = [onehot_quotes[k:k+2] for k in range(len(onehot_quotes)-1)]

part = int(0.8*len(samples))

train = samples[:part]
test = samples[part:]

np.save('train.npy', train)
np.save('test.npy', test)

vocab = {
            'word2id' :word2index,
            'id2word' : index2word ,
            'freq_dist' : freq_dist
                }

with open('data.pkl', 'wb') as f:
        pickle.dump(vocab, f)

