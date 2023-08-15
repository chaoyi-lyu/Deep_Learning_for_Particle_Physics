import csv
import numpy as np
import pandas as pd
import glob
from sklearn.utils import shuffle

class ReadCSVData:
    """ Read CSV datasets
    Args:
        
    """
    def __init__(self, file_paths):
        if isinstance(file_paths, list):
            dataset = pd.DataFrame([])
            for file_path in file_paths:
                data = self.read_data(file_path)
                dataset = pd.concat([dataset, data])
        elif isinstance(file_paths, str):
            dataset = self.read_data(file_paths)
        self.data = self.shuffle_data(dataset)
        
    def read_data(self, file_path):
        n_columns = 0
        with open(file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                n_columns = max(len(row), n_columns)
        self.n_columns = int(n_columns + (n_columns-1)/4 + 4) 
        column_name = ['event', 'process', 'weight', 'MET', 'METphi']
        for i in range(1, int((self.n_columns-5)/5+1)):
            column_name += [f'obj{i}',f'E{i}',f'pt{i}',f'eta{i}',f'phi{i}']
        return pd.read_csv(file_path, sep=',|;', engine='python', names=column_name+['nothing'], header=None).drop(['nothing'], axis=1)
        
    def labels(self, ):
        embedding = pd.get_dummies(self.data.process)
        return embedding.columns, embedding.values
    
    def objects(self, ):
        particle_columns = self.data.columns[self.data.columns.str.startswith('obj')]
        return self.data[particle_columns].values
    
    def kin(self, ):
        particle_columns = self.data.columns[self.data.columns.str.startswith(('E','pt','eta','phi'))]
        return self.data[particle_columns].fillna(0).values
    
    def shuffle_data(self, data):
        return shuffle(data)
