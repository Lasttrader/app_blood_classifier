# blood-classifier

dataset was build from https://www.cdc.gov/heartdisease/tools_training.htm

classification on heart disease


=======


ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Project Structure
This project has four major parts in app/:

model.py - This contains code fot our Machine Learning model to predict

app.py - This contains Flask APIs 

templates - This folder contains the HTML template

Running the project


This would create a serialized version of our model into a file KNN01.pkl

Run app.py using below command to start Flask API
python app.py

By default, flask will run on port 5000.

Navigate to URL http://localhost:5000

You should be able to view the homepage as below : alt text

Enter valid numerical values in all  input boxes and hit Predict.

If everything goes well, you should be able to see the predcited vaule on the HTML page

 