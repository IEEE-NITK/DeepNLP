from flask import Flask, render_template, request
from flask import jsonify

app = Flask(__name__,static_url_path="/static") 

#############
# Routing
#
# If we want to give a normal message as reply, use type='text' in the json response
# if we want a set of links specify type as links and provide an array of links in the json response
@app.route('/message', methods=['POST'])
def reply():
    return jsonify( { 'type' : 'text', 'text': 'Keep trucking on.' } )

@app.route('/links', methods=["POST"])
def returnLinks():
	return jsonify({'type':'links','links':['link1','link2','link3']})

@app.route("/")
def index(): 
    return render_template("index.html")


if (__name__ == "__main__"): 
    app.run(port = 5000) 
