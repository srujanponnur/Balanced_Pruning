from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
#Note: CIFAR10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse,
# ship, truck

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

model = load_model('/home/snaray23/591/cifar_conditional_generator_epochs.h5')

# generate multiple images

latent_points, labels = generate_latent_points(100, 750)
# specify labels - generate 10 sets of labels each gping from 0 to 9
labels = asarray([8 for x in range(750)])
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)


from PIL import Image as im
# print(x_fake[0].shape)
for i in range(750):
  fake_save = im.fromarray(X[i], 'RGB')
#   pyplot.imshow(X[i])
#   pyplot.show()
  fake_save.save(f"/home/snaray23/591/fake_ships/fake_planes{i}.jpeg")

# plot the result (10 sets of images, all images in a column should be of same class in the plot)
# Plot generated images 
# def show_plot(examples, n):
# 	for i in range(n * n):
# 		plt.subplot(n, n, 1 + i)
# 		plt.axis('off')
# 		plt.imshow(examples[i, :, :, :])
# 	plt.show()
    
# show_plot(X, 10)