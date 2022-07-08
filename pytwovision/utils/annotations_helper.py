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
        temp_data = np.asarray(self.annotations)
        temp_data = pd.DataFrame(temp_data)
        self.annotations = temp_data.sample(frac=1)
        self.annotations = self.annotations.values.tolist()
    
    def split(self, work_dir, train_percentage=0.8, random_state=25, export=True, train_name="train", test_name="test"):
        """
        Arguments:
            train_percentage: a float between (0, 1) that corresponds with train data proportion.
            random_state: int, array-like, BitGenerator, np.random.RandomState
            export: a boolean 
        """
        data = np.asarray(self.annotations)
        data = pd.DataFrame(data)
        train_data = data.sample(frac=train_percentage, random_state=random_state)
        test_data = data.drop(train_data.index)
        if export:
            if train_data.shape[0] != 0:
                train_data.to_csv(os.path.join(work_dir, "{}.txt".format(train_name)), header=None, index=None, sep='\t')
            else:
                print("train data len is 0")
            if test_data.shape[0] != 0:
                test_data.to_csv(os.path.join(work_dir, "{}.txt".format(test_name)), header=None, index=None, sep='\t')
            else:
                print("test data len is 0")

        return train_data, test_data
