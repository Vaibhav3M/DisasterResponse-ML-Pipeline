# Disaster Response Pipeline Project

The pipeline classifies the messages transmitted during time of natural disasters. Data from FigureEight.

The ETL is as below:
		Extract: Read and merge data from data/disaster_categories.csv and data/disaster_messages.csv.
        Transform: Normalize, Tokenize, Lemmatize the messages. Convert categories into usable form.   
        Load: Store data into an SQL database file.

ML pipeline:
		Build an NLP pipeline.
        Run GridSearch to hypertune parameters.
        Evaluate and store the model.

Deploy model on Flask App:
		Visualize the results using Plotly.
        Display results for entered values.
        

### Instructions to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
