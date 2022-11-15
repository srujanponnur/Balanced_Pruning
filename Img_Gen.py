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


###############################
# example of loading the generator model and generating images

 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# plot the generated images
def create_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()
 
# load model
model = load_model(f'/home/snaray23/591/generator_model.hdf5')
# generate images
latent_points = generate_latent_points(100, 1000)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
create_plot(X, 10)
print(X.shape)

from PIL import Image as im
# print(x_fake[0].shape)
for i in range(1000):
  temp = (X[i] + 1) / 2.0
  fake_save = im.fromarray(temp, 'RGB')
#   pyplot.imshow(X[i])
#   pyplot.show()
  fake_save.save(f"/home/snaray23/591/aug/fake{i}.jpeg")

# ############################
# # example of generating an image for a specific point in the latent space
# from keras.models import load_model
# from numpy import asarray
# from matplotlib import pyplot
# # load model
# model = load_model(f'/home/snaray23/591/generator_model.hdf5')
# # all 0s
# vector = asarray([[0.75 for _ in range(100)]])
# # generate image
# X = model.predict(vector)
# # scale from [-1,1] to [0,1]
# X = (X + 1) / 2.0
# # plot the result
# pyplot.imshow(X[0, :, :])
# pyplot.show()