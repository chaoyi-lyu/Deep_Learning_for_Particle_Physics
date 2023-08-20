import csv
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
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
        n_columns = 81
        with open(file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                n_columns = max(len(row), n_columns)
        self.n_columns = int( n_columns + n_columns/4 ) 
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
    
    def kin(self, log=True):
        particle_columns = self.data.columns[self.data.columns.str.startswith(('E','pt','eta','phi'))]
        kine = self.data[particle_columns].fillna(0).values
        if log:
            with np.errstate(divide='ignore'):
                kine[:,0::4] = np.log10(kine[:,0::4]/1000)
                kine[:,1::4] = np.log10(kine[:,1::4]/1000)
                kine[kine == -np.inf] = 0
        return kine
    
    def shuffle_data(self, data):
        return shuffle(data)

class DataLoader():
    def __init__(self, root_path):
        self.root_path = root_path
        self.folders = self.find_folder()

    def find_folder(self,):
        folders = [(self.root_path+i) for i in os.listdir(self.root_path) if os.path.isdir(self.root_path + i)]
        folders.remove(self.root_path+'min_20000')
        folders.remove(self.root_path+'large')
        return folders

    def read_file(self, path):
        file_size = sum(1 for line in open(path)) - 1
        sample_size = 29
        skip = sorted(random.sample(range(file_size), file_size - sample_size))
        column_name = ['event', 'process', 'weight', 'MET', 'METphi']
        for i in range(1, 20):
            column_name += [f'obj{i}',f'E{i}',f'pt{i}',f'eta{i}',f'phi{i}']
        df = pd.read_csv(path, skiprows=skip, sep=',|;', engine='python', names=column_name+['nothing'], header=None).drop(['nothing'], axis=1)
        return df 

    def load_dataframe(self, folders):
        df_data = pd.DataFrame([])
        for i in folders:
            files = glob.glob(i+'/*csv')
            file_path = files[np.random.randint(len(files))]
            df_data = pd.concat([df_data, self.read_file(file_path)])
        # for file_path in glob.glob(path + '*csv'):
        #     df_data = pd.concat([df_data, self.read_file(file_path)])
        return df_data

    def kin(self, df, log=True):
        particle_columns = df.columns[df.columns.str.startswith(('E','pt','eta','phi'))]
        kine = df[particle_columns].fillna(0).values
        if log:
            with np.errstate(divide='ignore'):
                kine[:,0::4] = np.log10(kine[:,0::4]/1000)
                kine[:,1::4] = np.log10(kine[:,1::4]/1000)
                kine[kine == -np.inf] = 0
        return kine

    def objects(self, df):
        particle_columns = df.columns[df.columns.str.startswith('obj')]
        return df[particle_columns].values

    def data_process(self, df):
        missing_obj = np.stack([np.zeros(len(df)), np.log10(df['MET'].values/1000), np.ones(len(df))*(-1), df['METphi'].values], axis=1)
        data = torch.concat([torch.tensor(missing_obj), torch.tensor(self.kin(df))], axis=1)
    
        objects = pd.DataFrame(self.objects(df)).fillna( np.nan ).values
        objects = np.concatenate([np.expand_dims(np.array(['M']*len(df)),axis=1), objects], axis=1)
        objects = np.vstack([np.array(['M','j','b','e-','e+','m-','m+','g',np.nan]*20, dtype='object').reshape(20,9).T, objects])
        objects = pd.DataFrame(objects)
        one_hot_encoded = pd.get_dummies(objects, dummy_na=True)[9:].values
    
        return data, torch.tensor(one_hot_encoded)

    def next(self, ):
        while True:
            df = self.load_dataframe(self.folders)
            yield self.data_process(df)
