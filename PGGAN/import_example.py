import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import sys




# Initialize TensorFlow session.
tf.InteractiveSession()
sys.path.append('/ssd1/U1_data/progressive_growing_of_gans-master/pkl/karras2018iclr-celebahq-1024x1024.pkl')
# Import official CelebA-HQ networks.
with open('pkl/karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
latent1 = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
for i in range(500):
	latents = latent1[[i]]
	# Generate dummy labels (not used by the official networks).
	labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

	# Run the generator to produce a set of images.
	images = Gs.run(latents, labels)

	# Convert images to PIL-compatible format.
	images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
	images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

	# Save images as PNG.
	PIL.Image.fromarray(images[0],'RGB').save('person/img%d.png' % i)
