"""
TODO
    Find way to vary convolution loops in config
    Figure out how to manage memory with higher resolution images

"""

# Source Materials
"""
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
https://github.com/jeffheaton/t81_558_deep_learning/blob/c591f3fea9c086407820c66e4c4a667e96f47f62/t81_558_class_07_2_Keras_gan.ipynb
https://www.youtube.com/watch?v=Wwwyr7cOBlU
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
https://www.tensorflow.org/tutorials/generative/style_transfer # documentation on wonderful things
https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9


*/
"""

# external libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# file where our network architectures are defined
import networks
import helpers

image_list = []

# Config
# -----------------------------------------------------------------------------------

FILE_PATH = 'four_shapes/shapes/circle'
SAVE_DIR = 'output/four_shapes/shapes/circle'

# FILE_PATH = 'four_shapes/shapes/star'
# SAVE_DIR = 'output/four_shapes/shapes/star'

# FILE_PATH = 'E:/NASA-Space-images'
# SAVE_DIR = 'output/space_images'

# FILE_PATH = 'four_shapes/shapes/'
# SAVE_DIR = 'output/four_shapes/shapes/all'

# FILE_PATH = 'E:/5857_1166105_bundle_archive/fruits-360/Test/Apple Braeburn'
# SAVE_DIR = 'output/fruits/apples'

# FILE_PATH = 'testing_images'
# SAVE_DIR = 'output/testing_images'

# FILE_PATH = 'flower_images/'

# FILE_PATH = 'E:/gemstones-images/train/All_Stones'
# FILE_PATH = 'E:/gemstones-images/train/Bixbite'
# SAVE_DIR = 'output/gemstones'

FILE_TYPES = ['jpg', 'png']

SHOW_OUTPUT = False
SAVE_OUTPUT = True

resize_shape = (128, 128)
SEED_SIZE = 64

SAVE_FREQ = 1
EPOCHS = 1000
BATCH_SIZE = 2
DISCRIMINATOR_RATIO = 4

discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=5e-5)
generator_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=5e-6)

PREVIEW_ROWS = 4
PREVIEW_COLS = 4
PREVIEW_MARGIN = 5

# -----------------------------------------------------------------------------------

width, height, depth, image_list = helpers.import_data(FILE_PATH, resize_shape, FILE_TYPES)

GENERATE_SQUARE = width
IMAGE_CHANNELS = depth

print("Image List Shape: ", image_list.shape)
print("Width of Image: ", width)
print("Height of Image: ", height)
print("Depth of image: ", depth)
print(f"Will generate {GENERATE_SQUARE}px square images.")

image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)

# imgplot = plt.imshow(image_list[0])
# plt.show()

generator = networks.build_generator(SEED_SIZE, image_shape)

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)

discriminator = networks.build_discriminator(image_shape, depth)
decision = discriminator(generated_image)

seed = tf.random.normal([16, SEED_SIZE])

train_data = tf.data.Dataset.from_tensor_slices(image_list).shuffle(30000).batch(BATCH_SIZE)

print("Training started")

helpers.train(train_data, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer,
              seed, BATCH_SIZE, SEED_SIZE, SAVE_FREQ, DISCRIMINATOR_RATIO, SHOW_OUTPUT, SAVE_DIR, SAVE_OUTPUT)

