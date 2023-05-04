# -*- coding: utf-8 -*-
"""UniMiB_SHAR_ADL_load_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U1EY6cZsOFERD3Df1HRqjuTq5bDUGH03

#UniMiB_SHAR_ADL_load_dataset.ipynb. 
Loads the A-9 (ADL) portion of the UniMiB dataset from the Internet repository and converts the data into numpy arrays while adhering to the general format of the [Keras MNIST load_data function](https://keras.io/api/datasets/mnist/#load_data-function).

Arguments: tbd
Returns: Tuple of Numpy arrays:   
(x_train, y_train),(x_validation, y_validation)\[optional\],(x_test, y_test) 

* x_train\/validation\/test: containing float64 with shapes (num_samples, 151, {3,4,1})
* y_train\/validation\/test: containing int8 with shapes (num_samples 0-9)

The train/test split is by subject

Example usage:  
x_train, y_train, x_test, y_test = unimib_load_dataset()

Additional References  
 If you use the dataset and/or code, please cite this paper (downloadable from [here](http://www.mdpi.com/2076-3417/7/10/1101/html))

Developed and tested using colab.research.google.com  
To save as .py version use File > Download .py

Author:  Lee B. Hinkle, IMICS Lab, Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


TODOs:
* Fix document strings
* Assign names to activities instead of numbers
"""

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
#from tensorflow.keras.utils import to_categorical # for one-hot encoding

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
from torch.utils.data import Dataset, DataLoader

#data augmentation
import tsaug

class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}

class unimib_load_dataset(Dataset):
    def __init__(self, 
        verbose = False,
        incl_xyz_accel = False, #include component accel_x/y/z in ____X data
        incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                               #False => combine test & validation groups
        is_normalize = False,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        one_hot_encode = True, data_mode = 'Train', single_class = False, class_name= 'Walking', augment_times = None):
        
        self.verbose = verbose
        self.incl_xyz_accel = incl_xyz_accel
        self.incl_rms_accel = incl_rms_accel
        self.incl_val_group = incl_val_group
        self.split_subj = split_subj
        self.one_hot_encode = one_hot_encode
        self.data_mode = data_mode
        self.class_name = class_name
        self.single_class = single_class
        self.is_normalize = is_normalize
        
        
        #Download and unzip original dataset
        if (not os.path.isfile('./UniMiB-SHAR.zip')):
            print("Downloading UniMiB-SHAR.zip file")
            #invoking the shell command fails when exported to .py file
            #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            self.download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
        if (not os.path.isdir('./UniMiB-SHAR')):
            shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
        #Convert .mat files to numpy ndarrays
        path_in = './UniMiB-SHAR/data'
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

        if(self.verbose):
            headers = ("Raw data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Reshape data and compute total (rms) acceleration
        num_samples = 151 
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        if (self.incl_rms_accel):
            rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
            adl_data = np.dstack((adl_data,rms_accel))
        #remove component accel if needed
        if (not self.incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
        if(verbose):
            headers = ("Reshaped data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html


        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels

        if (not self.incl_val_group):
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj'] + 
                                            self.split_subj['validation_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]
        else:
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]

            validation_index = np.nonzero(np.isin(sub_num, self.split_subj['validation_subj']))
            x_validation = adl_data[validation_index]
            y_validation = act_num[validation_index]

        test_index = np.nonzero(np.isin(sub_num, self.split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]

        if (verbose):
            print("x/y_train shape ",x_train.shape,y_train.shape)
            if (self.incl_val_group):
                print("x/y_validation shape ",x_validation.shape,y_validation.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)
        #If selected one-hot encode y_* using keras to_categorical, reference:
        #https://keras.io/api/utils/python_utils/#to_categorical-function and
        #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        if (self.one_hot_encode):
            y_train = self.to_categorical(y_train, num_classes=9)
            if (self.incl_val_group):
                y_validation = self.to_categorical(y_validation, num_classes=9)
            y_test = self.to_categorical(y_test, num_classes=9)
            if (verbose):
                print("After one-hot encoding")
                print("x/y_train shape ",x_train.shape,y_train.shape)
                if (self.incl_val_group):
                    print("x/y_validation shape ",x_validation.shape,y_validation.shape)
                print("x/y_test shape  ",x_test.shape,y_test.shape)
#         if (self.incl_val_group):
#             return x_train, y_train, x_validation, y_validation, x_test, y_test
#         else:
#             return x_train, y_train, x_test, y_test
        
        # reshape x_train, x_test data shape from (BH, length, channel) to (BH, channel, 1, length)
        self.x_train = np.transpose(x_train, (0, 2, 1))
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, self.x_train.shape[2])
        self.x_train = self.x_train[:,:,:,:-1]
        self.y_train = y_train
        
        self.x_test = np.transpose(x_test, (0, 2, 1))
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, self.x_test.shape[2])
        self.x_test = self.x_test[:,:,:,:-1]
        self.y_test = y_test
        print(f'x_train shape is {self.x_train.shape}, x_test shape is {self.x_test.shape}')
        print(f'y_train shape is {self.y_train.shape}, y_test shape is {self.y_test.shape}')
        
        
        if self.is_normalize:
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)
        
        #Return the give class train/test data & labels
        if self.single_class:
            one_class_train_data = []
            one_class_train_labels = []
            one_class_test_data = []
            one_class_test_labels = []

            for i, label in enumerate(y_train):
                if label == class_dict[self.class_name]:
                    one_class_train_data.append(self.x_train[i])
                    one_class_train_labels.append(label)

            for i, label in enumerate(y_test):
                if label == class_dict[self.class_name]:
                    one_class_test_data.append(self.x_test[i])
                    one_class_test_labels.append(label)
            self.one_class_train_data = np.array(one_class_train_data)
            self.one_class_train_labels = np.array(one_class_train_labels)
            self.one_class_test_data = np.array(one_class_test_data)
            self.one_class_test_labels = np.array(one_class_test_labels)
            
            if augment_times:
                augment_data = []
                augment_labels = []
                for data, label in zip(one_class_train_data, one_class_train_labels):
#                     print(data.shape) # C, 1, T
                    data = data.reshape(data.shape[0], data.shape[2]) # Channel, Timestep
                    data = np.transpose(data, (1, 0)) # T, C
                    data = np.asarray(data)
                    for i in range(augment_times):
                    
                        aug_data = tsaug.Quantize(n_levels=[10, 20, 30]).augment(data)
                        aug_data = tsaug.Drift(max_drift=(0.1, 0.5)).augment(aug_data)                   
#                         aug_data = my_augmenter(data) # T, C 
                        aug_data = np.transpose(aug_data, (1, 0)) # C, T
                        aug_data = aug_data.reshape(aug_data.shape[0], 1, aug_data.shape[1]) # C, 1, T
                        augment_data.append(aug_data)
                        augment_labels.append(label)
                
                augment_data = np.array(augment_data)
                augment_labels = np.array(augment_labels)
                print(f'augment_data shape is {augment_data.shape}')
                print(f'augment_labels shape is {augment_labels.shape}')
                self.one_class_train_data = np.concatenate((augment_data, self.one_class_train_data), axis = 0)
                self.one_class_train_labels = np.concatenate((augment_labels, self.one_class_train_labels), axis = 0)
            
            print(f'return single class data and labels, class is {self.class_name}')
            print(f'train_data shape is {self.one_class_train_data.shape}, test_data shape is {self.one_class_test_data.shape}')
            print(f'train label shape is {self.one_class_train_labels.shape}, test data shape is {self.one_class_test_labels.shape}')
        
    def download_url(self, url, save_path, chunk_size=128):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]       

    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,0,:] = self._normalize(epochs[i,j,0,:])
#                 epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])

        return epochs
    
    def __len__(self):
        
        if self.data_mode == 'Train':
            if self.single_class:
                return len(self.one_class_train_labels)
            else:
                return len(self.y_train)
        else:
            if self.single_class:
                return len(self.one_class_test_labels)
            else:
                return len(self.y_test)
        
    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            if self.single_class:
                return self.one_class_train_data[idx], self.one_class_train_labels[idx]
            else:
                return self.x_train[idx], self.y_train[idx]
        else:
            if self.single_class:
                return self.one_class_test_data[idx], self.one_class_test_labels[idx]
            else:
                return self.x_test[idx], self.y_test[idx]
            
    