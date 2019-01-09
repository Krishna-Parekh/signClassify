
'''
This file comtains the functions for generating separate datasets for training
 and validation. The function is generate_data(). It also prepares data for 
 final testing. The function to be called for it is generate_test().

'''


from tensorflow.keras.preprocessing import image
import os
from PIL import Image, ImageFile
#from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from google.cloud import storage


def generate_data(image_dir, image_width=100, image_height=100):
    
    #print "Hiiiii"
    directory = image_dir
    dataset_full = []
    print("directory", directory)
    
    folders = next(os.walk(directory))
    #print("os.walk(dir)", os.walk(directory))
    count_fol = 0
    count_file = 0
    for fol in folders[1]:
        fol_path = directory+'/' + fol + "/"
        count_fol += 1
        #print(os.walk(fol_path))
        #print("utils.py", next(os.walk(fol_path)))
        files = next(os.walk(fol_path))[2]
        
        for f in files:
            file_lbl = []
            file_path = fol_path + f
            file_lbl.append(file_path)
            file_lbl.append(fol)
            count_file += 1
            dataset_full.append(file_lbl)
    selector = np.random.random((count_file))
    
    #print(selector)
    
    dataset_train = []
    dataset_test = []
    
    for i in range(len(selector)):
        if selector[i] > 1.0/3.0:
            #print(selector[i], "train")
            selector[i] = 1   # For training
            dataset_train.append(dataset_full[i])
        else:
            selector[i] = 0   # For testing
            dataset_test.append(dataset_full[i])
    
    # Now the classification  
    train_images = []
    train_labels = []
    
    #print(selector)
    
    image_rows = image_height  # Size of the image
    image_cols = image_width
    
    le = LabelEncoder()
     
    print("dataset_train", len(dataset_train))
    print("dataset_test", len(dataset_test))
      
    for i in range(0, len(dataset_train)):
        img = image.load_img(dataset_train[i][0])
        img = img.resize((image_rows, image_cols), Image.ANTIALIAS)   # Resize fist two dimensions
        img_ar = np.array(img)
        lbl = dataset_train[i][1]
        train_images.append(img_ar)
        train_labels.append(lbl)

    train_images = np.array(train_images)
    train_labels = le.fit_transform(train_labels)    # Get numeric class labels
    
    test_images = []
    test_labels = ["" for _ in range(0, len(dataset_test))]
           
    for i in range(0, len(dataset_test)):
        img = image.load_img(dataset_test[i][0])
        img = img.resize((image_rows, image_cols), Image.ANTIALIAS)   # Resize fist two dimensions
        img_ar = np.array(img)
        lbl = dataset_test[i][1]
        test_images.append(img_ar)
        test_labels[i] = lbl

    test_images = np.array(test_images)
    len(test_images)
    test_labels = le.fit_transform(test_labels)
 
    return train_images, train_labels, test_images, test_labels



