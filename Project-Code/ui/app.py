from flask import Flask, render_template, request
from flask import jsonify
from .utils import word_vectors

app = Flask(__name__,static_url_path="/static") 

#############
# Routing
#
@app.route('/message', methods=['POST'])
def reply():
    return jsonify( { 'text': 'Keep trucking on.' } )

@app.route("/")
def index(): 
    return render_template("index.html")

@app.rout('/words/relationships', methods=['POST'])
def relationship():
	start1, end1, start2 = json.loads(request.form['data'])
	try:
		return word_vectors.relationship(start1,end1,start2)
	except:
		return jsonify({'error': 'word not in index'})

if (__name__ == "__main__"): 
    app.run(port = 5000) 
