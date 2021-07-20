import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import os

imPath='.\\images\\'
os.chdir('change to current directory')

readImage = load_img(imPath+'16.jpg')
pyplot.imshow(readImage)
readImage_np=np.array(readImage)

x_imageSet=np.empty((1,readImage_np.shape[0],readImage_np.shape[1],3))
x_imageSet[0]=readImage_np

# create image data augmentation generator
imageGen = ImageDataGenerator(rotation_range=90)
# prepare iterator
iter = imageGen.flow(x_imageSet, batch_size=1)

for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = iter.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()