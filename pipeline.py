from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

########################################################################
########################################################################
def identity(img): 
    return img

def mean(img, X=3):
  R = np.full((X,X), 1.0/(X**2.0), dtype=float)
  return ndimage.convolve(img, R)

def median(img, size=6):
  return ndimage.median_filter(img,size)

def gaussian(img, std=5):
  return ndimage.gaussian_filter(img,std)

def hp(img, c=3, d=3):
  R = np.full((d,d), -c, dtype=float)
  R[d/2,d/2] = 8*c
  return ndimage.convolve(img,R)

def gaussianHP(img, std=6):
  lpImg = ndimage.gaussian_filter(img,std)
  hpImg = img - lpImg
  return hpImg

filters = {"identity":identity,
           "mean":mean,
           "median":median,
           "gaussian":gaussian,
           "hp":hp,
           "gaussianHP":gaussianHP,
           }

########################################################################
########################################################################

def preprocess_batch(pipeline, imgs): 
    # Pipeline empty
    if len(pipeline) == 0:
        return imgs

    # Checking valid input for filter(s)
    for fn in pipeline:
        if fn not in filters:
            print("Error: Filter '%s' was not found in filters\n" % fn)
            exit(1)


    # # Applying filter(s) to each image
    new_imgs = []
    imgs = imgs.tolist()
    for img in imgs:
        fImg = img
        for fn in pipeline:
            f = filters[fn]
            fImg = f(fImg)
        
        new_imgs.append(fImg)

    return np.array(new_imgs)

########################################################################
########################################################################

PATH = "../cell_images/cell_images"
PROCESSED_PATH = "../cell_images/Processed"

class Dataset(object):
    def __init__(self, img_dir, imgs, name='training', verbose=True):

        self._dir = img_dir
        self._imgs = imgs
        self._infected_path = os.path.join(self._dir, 'Infected')
        self._uninfected_path = os.path.join(self._dir, 'Uninfected')

        self._num_infected = len(os.listdir(self._infected_path))
        self._num_uninfected = len(os.listdir(self._uninfected_path))
        self._total = self._num_infected + self._num_uninfected
        if verbose: 
            print("total ", name, " infected images: ", self._num_infected)
            print('total ', name, ' uninfected images: ', self._num_uninfected)
            print('Total ', name, ' images: ', self._total)

    def get_dir(self):
        return self._dir

    @property
    def num_infected(self): 
        return self._num_infected
        
    @property
    def num_uninfected(self): 
        return self._num_uninfected

    @property
    def total_examples(self):
        return self._total
    
    @property
    def imgs(self):
        return self._imgs 
    
########################################################################
########################################################################

class Datasets(object):
    def __init__(self, pipeline=['gaussianHP', 'gaussian'], load=True, save=False, num_examples=500):
        train_path = os.path.join(PATH, 'Train')
        val_path = os.path.join(PATH, 'Validation')

        new_train_imgs = None
        new_val_imgs = None

        if load: 
            # Preprocess Train images
            train_imgs = self._load_imgs('Train', num_examples)
            val_imgs = self._load_imgs('Validation', num_examples)
            new_train_imgs = self._preprocess(pipeline, train_imgs)
            if save: 
                train_path = os.path.join(PROCESSED_PATH, 'Train')
                self._save_imgs(new_train_imgs, train_path)

            # Preprocess Validation images
            new_val_imgs = self._preprocess(pipeline, val_imgs)
            if save: 
                val_path = os.path.join(PROCESSED_PATH, 'Validation')
                self._save_imgs(new_val_imgs, val_path)

        # Create training dataset
        self.train_data = Dataset(train_path, new_train_imgs, name='training')
        self.validation_data = Dataset(val_path, new_val_imgs, name='validation')

    @property 
    def train(self):
        return self.train_data
        
    @property 
    def validation(self):
        return self.validation_data

        # Create Validation dataset

    def _load_imgs(self, dirName, num_examples): 
        imgs = {'Infected': [], 'Uninfected': []}
        for c in imgs:
            path = os.path.join(PATH, dirName, c)
            for f in os.listdir(path): 
                if len(imgs[c]) == num_examples: break
                if f.endswith(".png"): 
                    imgs[c].append(mpimg.imread(os.path.join(path, f)))
            imgs[c] = np.array(imgs[c])
        return imgs

    def _preprocess(self, pipeline, imgs):
        """
        Preprocess images by running them through each filter in the pipeline. 
        """
        # Pipeline empty
        if len(pipeline) == 0:
            return imgs

        # Checking valid input for filter(s)
        for fn in pipeline:
            if fn not in filters:
                print("Error: Filter '%s' was not found in filters\n" % fn)
                exit(1)
        
        # Applying filter(s) to each image
        new_imgs = {'Infected': [], 'Uninfected': []}
        for c in imgs:
            for img in imgs[c]:
                fImg = img
                for fn in pipeline:
                    f = filters[fn]
                    fImg = f(fImg)
                
                new_imgs[c].append(fImg)

        return new_imgs

    def _save_imgs(self, new_imgs, path): 
        '''
        Only use this function if we want to save the preprocessed images
        and visualize them for later use (i.e. for report or understanding)
        '''
        # Delete Previously processed images
        try: 
            shutil.rmtree(path) 
        except: 
            pass
        # Create directory 
        # try:
        infectedPath = os.path.join(path, 'Infected')
        uninfectedPath = os.path.join(path, 'Uninfected')
        os.makedirs(infectedPath)
        os.mkdir(uninfectedPath)
        for c in new_imgs:
            for i, img in enumerate(new_imgs[c]):
                name = 'img' + str(i) + '.png'
                if c == 'Infected': path = os.path.join(infectedPath, name)
                else: path = os.path.join(uninfectedPath, name)
                mpimg.imsave(path, img)