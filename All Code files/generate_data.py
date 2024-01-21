from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class Generate_data():
    def __init__(self, datapath):
        """
        Generate_data class
        Two methods to be used
        1-split_test
        2-save_images
        [Note] that you have to split the public and private from fer2013 file
        """
        self.data_path = datapath

    def split_test(self, val_filename= 'val'):
        """
        Helper function to split the validation and train data from general train file.
            params:-
                data_path = path to the folder that contains the train data file
        """
        train_csv_path = self.data_path
        train = pd.read_csv(train_csv_path)
        validation_data = pd.DataFrame(train.iloc[3589:7178,:])
        train_data = pd.DataFrame(train.iloc[0:len(train)-7178,:])
        test_data = pd.DataFrame(train.iloc[7178:10767, :])
        print(train_data.shape)
        train_data.to_csv("data/train.csv")
        validation_data.to_csv("data/"+val_filename+".csv")
        test_data.to_csv("data/"+"test"+".csv")
        print("Done splitting the test file into validation & final test file")

    def str_to_image(self, str_img = ' '):
        '''
        Convert string pixels from the csv file into image object
            params:- take an image string
            return :- return PIL image object
        '''
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
        return Image.fromarray(imgarray)

    def save_images(self, datatype='train'):
        '''
        save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
            params:-
            datatype= str e.g (train, val, test)
        '''
        foldername= "data/"+datatype
        csvfile_path= "data/"+datatype+'.csv'
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        data = pd.read_csv(csvfile_path)
        images = data['pixels'] #dataframe to series pandas
        numberofimages = images.shape[0]
        for index in tqdm(range(numberofimages)):
            img = self.str_to_image(images[index])
            img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
        print('Done saving {} data'.format((foldername)))
