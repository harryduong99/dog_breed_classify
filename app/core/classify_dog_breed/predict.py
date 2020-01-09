from keras.models import load_model
import pickle

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

import os

class DogBreedPrediction:
    root_path = os.getcwd() + '/app/core/classify_dog_breed'

    def __init__(self, images):
        self.images = images

    def path_to_tensor(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def extract_VGG19(self, file_paths):
        tensors = self.paths_to_tensor(file_paths).astype('float32')
        preprocessed_input = preprocess_input_vgg19(tensors)
        return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

    def extract_Resnet50(self, file_paths):
        tensors = self.paths_to_tensor(file_paths).astype('float32')
        preprocessed_input = preprocess_input_resnet50(tensors)
        return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)
    
    def predict(self):
        breeds = []
        model = self.load_model()
        dogs_name = self.load_dog_name()
        # for image_path in self.images:
        # predict_img = [self.images]

        prd_vgg19 = self.extract_VGG19(self.images)
        prd_resnet50 = self.extract_Resnet50(self.images)

        prd_test = model.predict([prd_vgg19, prd_resnet50])  
        breed_prd_test = [np.argmax(prediction) for prediction in prd_test] 
        # breeds.append(dogs_name[breed_prd_test])
        
        for ind in breed_prd_test:
            breeds.append(dogs_name[ind])
        return breeds
        # return dogs_name[breed_prd_test[0]]

    def load_model(self):
        model_path = self.root_path + '/model/model_best.model'
        model_best = load_model(model_path)
        return model_best
    def load_dog_name(self):
        pickle_in = open(self.root_path + "/dog_names.pickle","rb")
        dog_names = pickle.load(pickle_in)
        return dog_names

# test_predict_img = root_path + "/predict_test/dog-1.jpg"
# test_predict_img = [test_predict_img]

# prd_vgg19 = extract_VGG19(test_predict_img)
# prd_resnet50 = extract_Resnet50(test_predict_img)

# prd_test = model_best.predict([prd_vgg19, prd_resnet50])  
# breed_prd_test = [np.argmax(prediction) for prediction in prd_test] 
# dog_names[breed_prd_test[0]]