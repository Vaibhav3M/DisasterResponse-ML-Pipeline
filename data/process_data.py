import sys

import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads data as from two csv files and merges them.
    
    INPUT: path of csv files
    OUTPUT: will return merged dataframe
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    df = messages_df.merge(categories_df, on='id', how='outer')
    
    return df

def clean_data(df):
    
    '''
    This function will clean and preprocess the data. 
    
    INPUT: dataframe to be cleaned
    OUTPUT: cleaned Dataframe
    '''

    #creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand =True)
    category_colnames =  re.sub(r"[^a-zA-Z;]", '',df['categories'][0]).split(';')
    categories.columns = category_colnames
    
    for column in categories:
        # converting column from string to numeric
        categories[column] = categories[column].str.replace('[a-zA-z-]','', regex=True).apply(lambda x: pd.to_numeric(x))
    
    # dropping the original categories column and appending new categories
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)          
                   
    #dropping duplicates records  
    df.drop_duplicates(subset=['id'], inplace =True)
    
    return df
    
    
def save_data(df, database_filename):
    '''
    This function saves the data to a SQL database file
    
    INPUT: Dataframe to be stored, Database name
    OUTPUT: None
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('FigureEightData', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()