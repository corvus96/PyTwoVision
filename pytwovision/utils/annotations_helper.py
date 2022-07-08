import numpy as np
import pandas as pd
import os

class AnnotationsHelper:
    def __init__(self, annotations_path):
        with open(annotations_path, 'r') as f:
            txt = f.readlines()
            self.annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        self.annotations_path = annotations_path

    def shuffle(self):
        """shuffle annotations"""
        temp_data = np.asarray(self.annotations)
        temp_data = pd.DataFrame(temp_data)
        self.annotations = temp_data.sample(frac=1)
        self.annotations = self.annotations.values.tolist()
    
    def export(self, data, file_path):
        """To export a dataframe like a .txt file
        Arguments:
            data: a pandas dataframe
            file_path: output file path
        """
        data.to_csv(file_path, header=None, index=None, sep='\t')
    
    def split(self, train_percentage=0.8, random_state=25):
        """Split annotations in two dataframes in reference of a percentage.
        Arguments:
            train_percentage: a float between (0, 1) that corresponds with train data proportion.
            random_state: int, array-like, BitGenerator, np.random.RandomState
        Returns:
            A tuple where the first element is train data and the second is test data
        """
        data = np.asarray(self.annotations)
        data = pd.DataFrame(data)
        train_data = data.sample(frac=train_percentage, random_state=random_state)
        test_data = data.drop(train_data.index)

        return train_data, test_data
