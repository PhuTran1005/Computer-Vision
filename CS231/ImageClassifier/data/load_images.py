import os

import numpy as np
from PIL import Image


class CatDogDataset:
    def __init__(self):
        pass
    
    def get_image_list(self, path):
        """
        Get list of image in a folder path
        """
        return os.listdir(path)
    
    def load_images(self, train_path, test_path, image_size):
        """
        Load all images
        """
        images_train_list = self.get_image_list(train_path)
        images_test_list = self.get_image_list(test_path)

        try:
            X_train = []
            y_train = []

            for i in range(len(images_train_list)):
                X_train.append(np.array(Image.open(os.path.join(train_path, images_train_list[i])).resize((image_size, image_size))).flatten())
                y_train.append(0 if images_train_list[i].split('.')[0] == 'cat' else 1)

        except:
            print('No have any train images')
            return
        
        try:
            X_test = []

            for i in range(len(images_test_list)):
                X_test.append(np.array(Image.open(os.path.join(test_path, images_test_list[i])).resize((image_size, image_size))).flatten())
        except:
            print('No have any test images')
            return
        
        return np.array(X_train), np.array(y_train), np.array(X_test)