import gensim.models.word2vec as w2v
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bhtsne import tsne
import os
import time

PATH = ""

# get pretrained word vectors trained  on the entire GoT text using word2vec in gensim
#thrones2vec = w2v.Word2Vec.load("thrones2vec.w2v")
thrones2vec = w2v.Word2Vec.load(os.path.join(PATH, "thrones2vec.w2v"))
all_word_vecs = thrones2vec.syn0
Y = np.load('tsne.npy')


points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, Y[thrones2vec.vocab[word].index])
            for word in thrones2vec.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


def best_avgs(words, all_vecs,k=10):
    
    
    
    ## get word embeddings for the words in our input array
    embs = np.array([thrones2vec[word] for word in words])
    #calculate its average
    avg = np.sum(embs,axis=0)/len(words)
    
    # Cosine Similarity with every word vector in the corpus
    denom = np.sqrt(np.sum(all_vecs*all_vecs,axis=1,keepdims=True)) \
            * np.sqrt(np.sum(avg*avg))
        
    similarity = all_vecs.dot(avg.T).reshape(all_vecs.shape[0],1) \
           / denom
    similarity = similarity.reshape(1,all_vecs.shape[0])[0]
    
    # Finding the 10 largest words with highest similarity
    # Since we are averaging we might end up getting the input words themselves 
    # among the top values
    # we need to make sure we get back len(words)+k closest words and then 
    # remove all input words we supplied
    
    nClosest = k + len(words)
    
    # Get indices of the most similar word vectors to our avgvector
    ind = np.argpartition(similarity, -(nClosest))[-nClosest:]
    
    names = [thrones2vec.index2word[indx] for indx in ind]
    similarity = similarity[ind]
    uniq = [(person,similar) for person,similar in zip(names,similarity) if person not in words]
    
          
    return sorted(uniq,key=itemgetter(1),reverse=True)[:k]


def relationship(start1, end1, start2):
	
    similarities = thrones2vec.most_similar_cosmul(positive=[start2, end1],negative=[start1])
    end2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return end2


def similarity(word):
	return thrones2vec.most_similar(word)



def coords(word):
    coord = points.loc[points["word"]==word,:].values[0]
    return coord[1],coord[2]



def plot_region(x_bounds, y_bounds):
    
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    inwords=[]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        inwords.append(point.word)
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    fig = ax.get_figure()
    file = 'IMG'+time.strftime("%Y%m%d%H%M%S")+'.png'
    fig.savefig(file)
    
    words = ", ".join(inwords)
    return (words,file)

def plot_close_to(word):
    x,y = coords(word)
    words,file = plot_region(x_bounds=(x-1.0,x+1.0), y_bounds=(y-1.0,y+1.0))
    return (words,file)

    
if __name__ == '__main__':

    words, file = plot_close_to("Winterfell")
    print(file)
    print(similarity("Arya"))
    print(relationship("man","woman","king"))