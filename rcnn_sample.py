import os 
import sys
import skimage
from skimage.io import imread
import dicom

DATA_DIR = '../pnemonia_data'
# Directory to save logs and trained model
ROOT_DIR = 'logdir'

example_image = '../pneumonia_data/test_images/000db696-cf54-4385-b10b-6b16fbb3f985.dcm'

im = imread(example_image)

skimage.imshow(im)