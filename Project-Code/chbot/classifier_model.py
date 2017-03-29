from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, TimeDistributed
from keras.layers import LSTM, Input, RepeatVector
from keras.preprocessing.sequence import pad_sequences
from scipy import spatial
import numpy as np
import nltk
import re
import sys
import pickle
from os.path import exists as file_exists
from functools import reduce

PREDICT_FLAG = False
TRAIN_FLAG = False

model_save_file = 'classifier1.h5'
input_qa_file = 'GOT_QA.pkl'
classifier_data_file = 'classifier_data.pkl'

pad_token = '<pad>'
loss_function = 'categorical_crossentropy'
optimizer = 'rmsprop'
metrics = ['accuracy']
maxlen = 10
num_epoch = 5
batch_size = 16
num_feat_word_vec = 128
num_feat_sent_vec = 128
NOT_A_WORD = -2

vocab = {}
vocab["word2id"] = {}
vocab["id2word"] = {}
is_word = re.compile(r'^[a-zA-Z]*$')

val_qas = pickle.load(open(input_qa_file, 'rb+'))
questions = list(map(lambda x: x[0], val_qas))
answers = list(map(lambda x: x[1], val_qas))

book_sents = pickle.load(open(classifier_data_file, 'rb+'))

def getID(word, create=True):
  if word != pad_token and not is_word.match(word):
    return NOT_A_WORD
  wid = vocab["word2id"].get(word, -1)
  if wid == -1:
    if create:
      wid = len(vocab["word2id"])
      vocab["word2id"][word] = wid
    else:
      wid = vocab["word2id"].get("<unknown>", NOT_A_WORD)
  return wid

def clean_input(questions, remove_non_vocab=False):
  lower_sent = lambda sent: [word.lower() for word in nltk.word_tokenize(sent)]
  questions = list(map(lambda x: lower_sent(x.strip()), questions))
  if remove_non_vocab:
    rem_non_words = lambda words: list(filter(lambda x: getID(x, create=False) != NOT_A_WORD, words))
    questions = list(map(lambda ques: rem_non_words(ques), questions))
  return list(map(lambda x: ' '.join(x), questions))

questions = clean_input(questions)
book_sents = clean_input(book_sents)


getID(pad_token)
for ques in questions:
  for word in nltk.word_tokenize(ques):
    getID(word)

for sent in book_sents:
  for word in nltk.word_tokenize(sent):
    getID(word)

vocab_length = len(vocab["word2id"])
vocab["id2word"] = { v: k for k, v in vocab["word2id"].items() }

id_mat = np.identity(2, dtype='int32')

print('Vocabulary created.')
print("Created vocabulary of " + str(vocab_length) + " words.")

def sen2enco(sentence):
  return [getID(word, create=False) for word in nltk.word_tokenize(sentence)]  

print('Creating training samples...')
onehot_quotes = [sen2enco(ques) for ques in questions]
book_onehotq = [sen2enco(ques) for ques in book_sents]

sequences_ques = pad_sequences(onehot_quotes, maxlen=maxlen, dtype='int32',
    padding='pre', truncating='pre', value=0.)
sequences_book_qs = pad_sequences(book_onehotq, maxlen=maxlen, dtype='int32',
    padding='pre', truncating='pre', value=0.)

labels = []
X_train = []

for x in sequences_ques:
  X_train.append(x)
  labels.append(id_mat[0])

for x in sequences_book_qs:
  X_train.append(x)
  labels.append(id_mat[1])

#labels = [list(map(lambda x: id_mat[x], y)) for y in sequences_ques]
labels = np.array(labels, dtype='int32')
X_train = np.array(X_train, dtype='int32')

input_dim = num_feat_word_vec
output_dim = 2

def create_seqauto_model():
  inputs = Input(shape=(maxlen,))
  embed = embeddding(vocab_length, input_dim, 
                    input_length=maxlen,
                    mask_zero=True)(inputs)
  encoded = LSTM(num_feat_sent_vec)(embed)
  dense_out = Dense(output_dim, activation='softmax')(encoded)
  sequence_autoencoder = Model(inputs, dense_out)
  return sequence_autoencoder

if file_exists(model_save_file):
  sequence_autoencoder = load_model(model_save_file)
else:
  sequence_autoencoder = create_seqauto_model()


sequence_autoencoder.compile(loss=loss_function,
                             optimizer=optimizer,
                             metrics=metrics)

def format_model_i(ques, remove_non_vocab=False):
  ques = re.sub(r'\?', '', ques)
  ques = clean_input([ques], remove_non_vocab)[0]
  id_s = sen2enco(ques)
  padded_ids = pad_sequences([id_s], maxlen=maxlen, dtype='int32',
    padding='pre', truncating='pre', value=0.)
  enc_ques = padded_ids
  return enc_ques

def format_model_o(pred):
  pred_l = np.argmax(pred, axis=1)[0]
  return pred_l

def predict(model, ques, remove_non_vocab=False):
  enc_ques = format_model_i(ques, remove_non_vocab)
  pred = model.predict(enc_ques)
  return format_model_o(pred)

def q2c_classify(ques):
  prediction_class = predict(sequence_autoencoder, ques, True)
  if prediction_class == 1:
    return "generative"
  elif prediction_class == 0:
    return "retrieval"
  else:
    return "somethingwentwrong"

def setup_flags():
  if sys.argv[1] == '--predict':
    PREDICT_FLAG = True
  elif sys.argv[1] == '--train':
    TRAIN_FLAG = True

def run_model():
  while PREDICT_FLAG and True:
    input_q = input("enter query: ")
    print(predict(sequence_autoencoder, input_q))

  q = 'y'
  while TRAIN_FLAG and q != 'n':
    sequence_autoencoder.fit(X_train, labels, batch_size=batch_size, nb_epoch=num_epoch)
    sequence_autoencoder.save(model_save_file)
    q = input("More? y/n: ")

if __name__ == '__main__':
  setup_flags()
  run_model()