import sys
from flask import Flask, request, jsonify


app = Flask(__name__)

# Define the root endpoint for GET requests
@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to a Simple Flask API!</h1>"

# Define an endpoint for GET requests
@app.route('/sqa', methods=['GET'])
def greetSQA():
    return "<h1>Welcome to the SQA course!</h1>"

'''
SSP Endpoint
Addition by Chris Hinkson @cmh02

This endpoint will display the text "Secure Software Process".
'''
@app.route('/ssp', methods=['GET'])
def endpoint_SSP():
     
	# Just return the text
	return "<h1>Secure Software Process</h1>"

'''
VANITY Endpoint
Addition by Chris Hinkson @cmh02

This endpoint will display my name.
'''
@app.route('/vanity', methods=['GET'])
def endpoint_VANITY():
    
	# Just return my name
    return "<h1>Chris Hinkson</h1>"

'''
MYPYTHON Endpoint
Addition by Chris Hinkson @cmh02

This endpoint will display the python version installed on my computer.
'''
@app.route('/mypython', methods=['GET'])
def endpoint_MYPYTHON():
     
	# Get python version with sys
	pythonVersion = sys.version
     
	# Return the python version as string
	return f"<h1>{pythonVersion}</h1>"

'''
CSSE Endpoint
Addition by Chris Hinkson @cmh02

This endpoint will display the text "Department of Computer Science and Software Engineering".
'''
@app.route('/csse', methods=['GET'])
def endpoint_CSSE():
	 
	# Just return the text
	return "<h1>Department of Computer Science and Software Engineering</h1>"


if __name__ == '__main__':
    app.run(debug=True)
