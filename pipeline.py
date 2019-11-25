from scipy import ndimage
from scipy.misc import imsave
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

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

def gaussianHP(img, std=5):
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

PATH = "../cell_images/"

class Datasets(object):
    def __init__(self, pipeline=['identity']):
        # Train images
        train_imgs = self._load_imgs('Train')
        new_imgs = self._preprocess(pipeline, train_imgs)
        self._save_imgs(new_imgs)

        # Validation images
        val_imgs = self._load_imgs('Validation')
        # self._preprocess()
        # self._save_imgs()

    def _load_imgs(self, dirName): 
        imgs = {'Infected': [], 'Uninfected': []}
        for c in imgs:
            path = os.path.join(PATH, dirName, c)
            for f in os.listdir(path): 
                if f.endswith(".png"): 
                    imgs[c].append(mpimg.imread(os.path.join(path, f)))
            imgs[c] = np.array(imgs[c])
        return imgs

    def _preprocess(self, pipeline, imgs):
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
                break

        return new_imgs
                

    def _save_imgs(self, new_imgs): 
        # Delete Previously processed images
        try: 
           shutil.rmtree('../cell_images/Processed') 
        # Create directory 
        try: 
            infectedPath = '../cell_images/Processed/Infected'
            uninfectedPath = '../cell_images/Processed/Uninfected'
            os.makedirs(infectedPath)
            os.mkdir(uninfectedPath)
        for c in new_imgs:
            for i, img in enumerate(new_imgs[c]):
                name = 'img' + str(i) + '.png'
                if c == 'Infected': path = os.path.join(infectedPath, name)
                else: path = os.path.join(uninfectedPath, name)
                mpimg.imsave(path, img)

d = Datasets()

########################################################################
########################################################################

class Dataset(object):
    def __init__(self, img_dir, name='training', verbose=True):


        self._dir = os.path.join(PATH, img_dir)
        self._infected_path = os.path.join(self._dir, 'Infected')
        self._uninfected_path = os.path.join(self._dir, 'Uninfected')

        self._num_infected = len(os.listdir(self._infected_path))
        self._num_uninfected = len(os.listdir(self._uninfected_path))
        self._total = self._num_infected + self._num_uninfected

        if verbose: 
            print('total ', name, ' infected images: ', self._num_infected)
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
    
