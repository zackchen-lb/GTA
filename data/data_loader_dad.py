import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


class NASA_Anomaly(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='SMAP', 
                 target=0, scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 8*60
            self.label_len = 2*60
            self.pred_len = 2*60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']

        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.flag = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def get_data_dim(self, dataset):
        if dataset == 'SMAP':
            return 25
        elif dataset == 'MSL':
            return 55
        elif str(dataset).startswith('machine'):
            return 38
        else:
            raise ValueError('unknown dataset '+str(dataset))

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """
        
        x_dim = self.get_data_dim(self.data_path)
        if self.flag == 'train':
            f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_path)), "rb")
            data = pickle.load(f).reshape((-1, x_dim))
            f.close()
        elif self.flag in ['val', 'test']:
            try:
                f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_path)), "rb")
                data = pickle.load(f).reshape((-1, x_dim))
                f.close()
            except (KeyError, FileNotFoundError):
                data = None
            try:
                f = open(os.path.join(self.root_path, self.data_path, '{}_test_label.pkl'.format(self.data_path)), "rb")
                label = pickle.load(f).reshape((-1))
                f.close()
            except (KeyError, FileNotFoundError):
                label = None
            assert len(data) == len(label), "length of test data shoube the same as label"
        if self.scale:
            data = self.preprocess(data)

        df_stamp = pd.DataFrame(columns=['date'])
        date = pd.date_range(start='1/1/2015', periods=len(data), freq='4s')
        df_stamp['date'] = date
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x:x//10)
        df_stamp['second'] = df_stamp.date.apply(lambda row:row.second,1)
        data_stamp = df_stamp.drop(['date'],1).values

        if self.flag == 'train':
            if self.features=='M':
                self.data_x = data
                self.data_y = data
            elif self.features=='S':
                df_data = data[:, [self.target]]
                self.data_x = df_data
                self.data_y = df_data
        else:
            border1s = [0, 0, 0]
            border2s = [None, len(data)//4, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            if self.features=='M':
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.label = label[border1:border2]
            elif self.features=='S':
                df_data = data[:, [self.target]]
                self.data_x = df_data[border1:border2]
                self.data_y = df_data[border1:border2]
                self.label = label[border1:border2]
        self.data_stamp = data_stamp
        
    def preprocess(self, df):
        """returns normalized and standardized data.
        """

        df = np.asarray(df, dtype=np.float32)

        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

        if np.any(sum(np.isnan(df)) != 0):
            print('Data contains null values. Will be replaced with 0')
            df = np.nan_to_num()

        # normalize data
        df = MinMaxScaler().fit_transform(df)
        print('Data normalized')

        return df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.flag == 'train':
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            seq_label = self.label[s_end:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class WADI(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='WADI_14days_downsampled.csv', 
                 target='1_AIT_001_PV', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 8*60
            self.label_len = 2*60
            self.pred_len = 2*60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = MinMaxScaler()
        if self.flag == 'train':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            'WADI_14days_downsampled.csv'))
            if self.features=='M':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

            df_stamp = df_raw[['date']]
            
            if self.scale:
                data = scaler.fit_transform(df_data.values)
            else:
                data = df_data.values
            
            self.data_x = data
            self.data_y = data
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            'WADI_attackdata_downsampled.csv'))

            border1s = [0, 0, 0]
            border2s = [None, len(df_raw)//4, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            df_stamp = df_raw[['date']][border1:border2]

            if self.features=='M':
                cols_data = df_raw.columns[1:-1]
                df_data = df_raw[cols_data]
                label = df_raw['label'].values
            elif self.features=='S':
                df_data = df_raw[[self.target]]
                label = df_raw['label'].values

            if self.scale:
                data = scaler.fit_transform(df_data.values)
            else:
                data = df_data.values
            
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.label = label[border1:border2]
        
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x:x//10)
        df_stamp['second'] = df_stamp.date.apply(lambda row:row.second,1)
        df_stamp['second'] = df_stamp.second.map(lambda x:x//10)
        data_stamp = df_stamp.drop(['date'],1).values
        
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.flag == 'train':
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            seq_label = self.label[s_end:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class SWaT(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='SWaT_normaldata_downsampled.csv', 
                 target='FIT_101', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 8*60
            self.label_len = 2*60
            self.pred_len = 2*60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = MinMaxScaler()
        if self.flag == 'train':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            'SWaT_normaldata_downsampled.csv'))
            if self.features=='M':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

            df_stamp = df_raw[[' Timestamp']]
            
            if self.scale:
                data = scaler.fit_transform(df_data.values)
            else:
                data = df_data.values
            
            self.data_x = data
            self.data_y = data
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            'SWaT_attackdata_downsampled.csv'))

            border1s = [0, 0, 0]
            border2s = [None, len(df_raw)//4, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            df_stamp = df_raw[[' Timestamp']][border1:border2]

            if self.features=='M':
                cols_data = df_raw.columns[1:-1]
                df_data = df_raw[cols_data]
                label = df_raw['Normal/Attack'].values
            elif self.features=='S':
                df_data = df_raw[[self.target]]
                label = df_raw['Normal/Attack'].values

            if self.scale:
                data = scaler.fit_transform(df_data.values)
            else:
                data = df_data.values
            
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.label = label[border1:border2]
        
        df_stamp[' Timestamp'] = pd.to_datetime(df_stamp[' Timestamp'])
        df_stamp['month'] = df_stamp[' Timestamp'].apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp[' Timestamp'].apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp[' Timestamp'].apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp[' Timestamp'].apply(lambda row:row.hour,1)
        df_stamp['minute'] = df_stamp[' Timestamp'].apply(lambda row:row.minute,1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x:x//10)
        df_stamp['second'] = df_stamp[' Timestamp'].apply(lambda row:row.second,1)
        df_stamp['second'] = df_stamp.second.map(lambda x:x//10)
        data_stamp = df_stamp.drop([' Timestamp'],1).values
        
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.flag == 'train':
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            seq_label = self.label[s_end:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1



if __name__ == '__main__':
    flag = 'test'
    dataset = NASA_Anomaly(root_path='./data/', data_path='MSL', flag=flag, size=(60, 30, 1))
    print(flag, len(dataset))
    # data_loader = DataLoader(
    #         dataset,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=2,
    #         drop_last=True)
    # for (x, y, x_stamp, y_stamp, label) in data_loader:
    #     print(x.size(), y.size(), x_stamp.size(), y_stamp.size(), label.size())
    #     break