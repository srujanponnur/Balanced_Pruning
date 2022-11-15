from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import np_utils, to_categorical


import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Input, Flatten, Embedding, multiply, Dropout
from keras.layers import Concatenate, GaussianNoise,Activation
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras import initializers
from keras import backend as K

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

model = load_model(f'/home/snaray23/591/cgan_model.hdf5')
samples = 100
latent_dim = 100
z = np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
labels = to_categorical(np.arange(0, 10).reshape(-1, 1), num_classes=10)
        
x_fake = model.predict([z, labels])
x_fake = np.clip(x_fake, -1, 1)
x_fake = (x_fake + 1) * 127
x_fake = np.round(x_fake).astype('uint8')

for k in range(samples):
    plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
    plt.imshow(x_fake[k])
    plt.title(class_names[k])

plt.tight_layout()
plt.show()