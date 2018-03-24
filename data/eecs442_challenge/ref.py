import numpy as np
import pickle
import h5py
from scipy.misc import imread
import os

data_dir = '/eecs442_challenge'

assert os.path.exists(data_dir)

def initialize(opt):
    return

def load_image(idx):
    p = os.path.join(data_dir, 'train', 'color', str(idx) + '.png')
    return imread(p,mode='RGB')

def load_mask(idx):
    p = os.path.join(data_dir, 'train', 'mask', str(idx) + '.png')
    return imread(p,mode='L')

def load_gt(idx):
    p = os.path.join(data_dir, 'train', 'normal', str(idx) + '.png')
    return imread(p,mode='RGB')

def setup_val_split(opt = None):
    train = range(20000)
    valid = []
    return train, np.array(valid)
