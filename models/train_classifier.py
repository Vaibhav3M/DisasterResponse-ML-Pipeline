import sys
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_data(database_filepath):
    '''
    This function loads data from given sql path.
    
    INPUT: SQL Database path
    OUTPUT: data in variables X,y and Category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('FigureEightData', engine)
    X = df['message'].values
    y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    
    return X,y,category_names


def tokenize(text):
    '''
    This method performs following text preprocessing
    - Normalization 
    - Tokenization
    - Lemmatization
        
    INPUT:
    text: message for tokenization
    
    OUTPUT:
    clean_tokens: wordlist after tokenization
    '''
    
    text=re.sub(r"[^a-zA-z0-9]"," ",text.lower())
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).strip())
       
    return clean_tokens


def build_model():
    '''
    This function builds pipline model, runs GridSearch to find best model.
    
    
    INPUT: None
    OUTPUT: returns best pipeline model
    '''
    #using LinearSVC as it is better for text classification
    pipeline = Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 123))))
            ])
    
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 5]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
        

def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function evaluates the model.
   
   INPUT: model object, X_test, y_test and category_names
   OUTPUT: prints classification report and gives multiple target accuracy
    '''
    
    y_pred = model.predict(X_test)
    
    class_report = classification_report(np.hstack(y_test), np.hstack(y_pred))
    accuracy = (y_pred == y_test).mean()

    print("Classification report:\n", class_report)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    This function saves the model to given path.
    
    INPUT: model object and file name or path (.pkl)
    OUTPUT: returns bool as model save status
    '''
    try:
        dump(model, model_filepath) 
        return True
    except:
        return False
        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()