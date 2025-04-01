# Data loading module responsible for loading the dataset.

import pandas as pd

def load_data(train_path: str, test_path: str):
    # Load training and test datasets.
    # Parameters:
    #     train_path (str): Path to training CSV file
    #     test_path (str): Path to test CSV file
    # Returns:
    #     train (pd.DataFrame): Training dataset
    #     test (pd.DataFrame): Test dataset
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Drop the unused ID field (a unique identifier with no value for modeling)
    train_ID = train['Id']
    test_ID = test['Id']
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)
    return train, test, train_ID, test_ID