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

model_save_file = 'seq_emb_8085model.h5'
input_qa_file = 'GOT_QA.pkl'
pad_token = '<pad>'
loss_function = 'categorical_crossentropy'
optimizer = 'rmsprop'
metrics = ['accuracy']
maxlen = 10
num_epoch = 30
batch_size = 16
num_feat_word_vec = 128
num_feat_sent_vec = 128
NOT_A_WORD = -2

PREDICT_FLAG = False
TRAIN_FLAG = False

vocab = {}
vocab["word2id"] = {}
vocab["id2word"] = {}
is_word = re.compile(r'^[a-zA-Z]*$')

# remove_sp = lambda x: re.sub(r"\.|\,|\:|\(|\)|\'", "", x)
# with open(input_file, 'r') as f:
#   questions = f.readlines()

val_qas = pickle.load(open(input_qa_file, 'rb+'))
questions = list(map(lambda x: x[0], val_qas))
answers = list(map(lambda x: x[1], val_qas))

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

# def load_all_into_vocab(questions):
getID(pad_token)
for ques in questions:
  for word in nltk.word_tokenize(ques):
    getID(word)

vocab_length = len(vocab["word2id"])
vocab["id2word"] = { v: k for k, v in vocab["word2id"].items() }

id_mat = np.identity(vocab_length, dtype='int32')

print('Vocabulary created.')
print("Created vocabulary of " + str(vocab_length) + " words.")

def sen2enco(sentence):
  return [getID(word, create=False) for word in nltk.word_tokenize(sentence)]  

print('Creating training samples...')
onehot_quotes = [sen2enco(ques) for ques in questions]

sequences_ques = pad_sequences(onehot_quotes, maxlen=maxlen, dtype='int32',
    padding='pre', truncating='pre', value=0.)

labels = [list(map(lambda x: id_mat[x], y)) for y in sequences_ques]
labels = np.array(labels, dtype='int32')

X_train = sequences_ques

input_dim = num_feat_word_vec
output_dim = vocab_length

def create_seqauto_model():
  inputs = Input(shape=(maxlen,))
  embed = Embedding(vocab_length, input_dim, 
                    input_length=maxlen,
                    mask_zero=True)(inputs)
  encoded = LSTM(num_feat_sent_vec)(embed)
  repeat_vec = RepeatVector(maxlen)(encoded)
  decoded = LSTM(input_dim, return_sequences=True)(repeat_vec)
  dense_out = Dense(output_dim, activation='softmax')
  classify = TimeDistributed(dense_out, input_shape=(maxlen, input_dim))(decoded)
  sequence_autoencoder = Model(inputs, classify)
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
  pred_l = np.argmax(pred, axis=2)[0]
  pred_l = list(map(lambda x: vocab["id2word"][x], pred_l))
  return ' '.join(list(filter(lambda x: x != pad_token, pred_l)))

def predict(model, ques):
  enc_ques = format_model_i(ques)
  pred = model.predict(enc_ques)
  return format_model_o(pred)

i_m = Model(input=sequence_autoencoder.input,
            output=sequence_autoencoder.get_layer('lstm_1').output)

stop_words = ['is','who','what','are','of','the']
s_e = lambda x,y: list(set(nltk.word_tokenize(x)).intersection(nltk.word_tokenize(y)))
s_c = lambda x,y: len(list(filter(lambda x: x not in stop_words, list(s_e(x,y)))))

def most_similar(ques, remove_non_vocab=False):
  preds = i_m.predict(X_train)
  enc_ques = format_model_i(ques, remove_non_vocab)
  ques = clean_input([ques], remove_non_vocab)[0]
  us_q_pr = i_m.predict(enc_ques)[0]
  cos_sim = lambda x,y: 1 - spatial.distance.cosine(x,y)
  reducer = lambda x,y: y if y[1] >= x[1] else x
  sim_lister = lambda q,s: [(i,(s_c(questions[i],s)+1 * cos_sim(preds[i],q))) for i in range(len(preds))]
  sim_list = sim_lister(us_q_pr,ques)
  msq = reduce(reducer, sim_list, sim_list[0])
  clean_answ = lambda x: re.sub('\[[0-9]\]', '', x)
  return clean_answ(answers[msq[0]])

def q2a_retrieval(ques):
  return most_similar(ques, remove_non_vocab=True)

def set_up_flags():
  if sys.argv[1] == '--predict':
    PREDICT_FLAG = True
  elif sys.argv[1] == '--train':
    TRAIN_FLAG = True

def run_model():
  while PREDICT_FLAG and True:
    input_q = input("enter query: ")
    print(most_similar(input_q))

  q = 'y'
  while TRAIN_FLAG and q != "n":
    sequence_autoencoder.fit(X_train, labels, batch_size=batch_size, nb_epoch=num_epoch)
    sequence_autoencoder.save(model_save_file)
    q = input("More? y/n: ")

if __name__ == "__main__":
  set_up_flags()
  run_model()
