# Disaster Response Pipeline

## A summary of the project:

This project contains real messages that were sent during disaster events. A machine learning pipeline was used to categorize these events so that the appropriate disaster relief agency can be alerted of the emergency situation.

## How to run the Python scripts:

1. To clean the data and store in a database, type the following in terminal:  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To train the classifier and save it, type the following in the terminal: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## To run the web-app: 

1. type the following in your terminal: `python run.py` 
2. then type the following: `env|grep WORK`
3. In a new web browser type: 'https://SPACEID-3001.SPACEDOMAIN' using the spaceid and domain that you received in step 2.

## An explanation of the files in the repository:
The repository includes two databases, one of disaster categories, and one of disaster messages, as csv. files

The repository includes a process_data.py file for data cleaning and processing, a train_classifier.py file for training the model, and finally a run.py file for running the web app.
