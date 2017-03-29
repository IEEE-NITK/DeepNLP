from flask import Flask, render_template, request, json, send_from_directory, jsonify
import socket
import sys
import subprocess
import tensorflow as tf
from retrieval_model import q2a_retrieval
#from generative_model import q2a_generative
from classifier_model import q2c_classify
#import chatbot
import re
import subprocess
import word_vectors
from textblob import TextBlob
from word_vectors import best_avgs
from word_vectors import relationship as w2v_relationship

def q2a_generative(query):
    #sess = tf.Session()
    #cb = chatbot.Chatbot()
    #sess, model = cb.set_up_things(sess)
    #return cb.get_answer(sess, model, query)
    a = subprocess.check_output(['python3', 'main.py', '_'.join(query.split())])
    b = a.decode().split('\n')[-2]
    return re.sub(r'\xe2\x80\x99', '\'', b)

def get_answer(query):
    query_class = q2c_classify(query)
    if query_class == 'retrieval':
        try:
            output = q2a_retrieval(query)
            return output
        except e:
            print(e)
    elif query_class == 'generative':
        try:
            output = q2a_generative(query)
            return output
        except e:
            print(e)
    return "Well, would you believe it! I have nothing to say."

def get_np(query):
    blob = TextBlob(query)
    nps = list(filter(lambda x: x not in ['who','what'], blob.noun_phrases))
    caps = list(map(lambda x: x.capitalize(), (' '.join(nps)).split()))
    return caps

def get_links(query):
    query_class = q2c_classify(query)
    if query_class == 'retrieval':
        return list(map(lambda x: x[0], best_avgs(get_np(query))))[:3]
    else:
        return False
app = Flask(__name__)

@app.route('/')
def submit():
    return render_template('index.html')

@app.route('/message', methods=['POST'])
def execute():
    query = request.form['msg']
    print(query)
    link_gen = get_links(query)
    if link_gen:
        return jsonify( { 'text' : get_answer(query), 'links': link_gen, 'type' : 'links'})
    else:
        return jsonify( { 'text' : get_answer(query) })

@app.route('/relationships', methods=['POST'])
def relationship():
    names = request.form['data']
    entit_arr = names.split('__')
    start1 = entit_arr[0]
    end1 = entit_arr[1]
    start2 = entit_arr[2]
    print(names)
    print("aailaa")
    try:
        return jsonify({'text' : w2v_relationship(start1,end1,start2)})
    except:
        return jsonify({'error': 'word not in index'})
# @app.route('/execute', methods=['POST'])
# def execute():
#     query = request.json['query']
#     output = ""
#     query_class = q2c_classify(query)
#     if query_class == 'retrieval':
#         try:
#             output = q2a_retrieval(query)
#         except e:
#             print(e)
#     elif query_class == 'generative':
#         output = "wow, you asked me a generative question"
#     return json.dumps({'result' : output})

if __name__ =="__main__":
    app.run()
