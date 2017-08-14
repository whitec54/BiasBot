from flask import Flask, jsonify, render_template, request, json
from baisbot import BiasBot

app = Flask(__name__)

#default page
@app.route('/')
def index():
	print("happened")
	return render_template('index.html')

@app.route('/_add_numbers')
def add_numbers():
	bb = BiasBot()

	url = request.args.get('url')
	print(bb.getDisplayData(url))

	url = "done did it"

	return jsonify(result=url)

if __name__ == "__main__":
	app.run(debug=True)
 