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
https://www.kaggle.com/jesucristo/gan-introduction
*/
"""

# external libraries
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# file where our network architectures are defined
import networks
import helpers

image_list = []

# Config
# -----------------------------------------------------------------------------------
#
FILE_PATH = 'four_shapes\\shapes\\circle'
SAVE_DIR = 'output\\four_shapes\\shapes\\circle'

# FILE_PATH = 'four_shapes\\shapes\\star'
# SAVE_DIR = 'output\\four_shapes\\shapes\\star'

# FILE_PATH = 'E:\\NASA-Space-images'
# SAVE_DIR = 'output\\space_images'

# FILE_PATH = 'four_shapes\\shapes\\'
# SAVE_DIR = 'output\\four_shapes\\shapes\\all'

# FILE_PATH = 'E:\\5857_1166105_bundle_archive\\fruits-360\\Test\\Apple Braeburn'
# SAVE_DIR = 'output\\fruits\\apples'
#
# FILE_PATH = 'testing_images'
# SAVE_DIR = 'output\\testing_images'

# FILE_PATH = 'flower_images\\'

# FILE_PATH = 'E:\\gemstones-images\\train\\All_Stones'
# FILE_PATH = 'E:\\gemstones-images\\train\\Bixbite'
# SAVE_DIR = 'output\\gemstones'

# FILE_PATH = 'E:\\Paintings\\resized\\128p'
# FILE_PATH = 'E:\Paintings\images\images\Pablo_Picasso'
# SAVE_DIR = 'output\\paintings'

# FILE_PATH = 'E:\\cats_set'
# SAVE_DIR = 'output\\cats'

FILE_TYPES = ['jpg', 'png']

SHOW_OUTPUT = False
SAVE_OUTPUT = True

resize_shape = (32, 32)
SEED_SIZE = 100

SAVE_FREQ = 5
EPOCHS = 500
BATCH_SIZE = 4
DISCRIMINATOR_RATIO = 2

generator_lr = 5e-5
discriminator_lr = 5e-5

PREVIEW_ROWS = 4
PREVIEW_COLS = 4
PREVIEW_MARGIN = 5

# -----------------------------------------------------------------------------------

reset = None
while reset is None and os.path.exists(SAVE_DIR + '\\Generator') and os.path.exists(SAVE_DIR + '\\Discriminator'):
    user_input = input("RESET or RESUME training?   ").lower()
    if user_input == 'reset':
        reset = True
    elif user_input == 'resume':
        reset = False

if reset is None:
    reset = 'reset'

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

generator = None
if os.path.exists(SAVE_DIR + '\\Generator') and not reset:
    generator = tf.keras.models.load_model(SAVE_DIR + '\\Generator')
else:
    generator = networks.build_generator(SEED_SIZE, image_shape)

discriminator = None

if os.path.exists(SAVE_DIR + '\\Discriminator') and not reset:
    discriminator = tf.keras.models.load_model(SAVE_DIR + '\\Discriminator')
else:
    discriminator = networks.build_discriminator(image_shape, depth)


starting_epoch = 0

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)
decision = discriminator(generated_image)

seed = tf.random.normal([16, SEED_SIZE])

train_data = tf.data.Dataset.from_tensor_slices(image_list).shuffle(30000).batch(BATCH_SIZE)

print("Training started")

helpers.train(train_data, starting_epoch, EPOCHS, generator, discriminator, seed, BATCH_SIZE, SEED_SIZE, SAVE_FREQ,
              DISCRIMINATOR_RATIO, SHOW_OUTPUT, SAVE_DIR, SAVE_OUTPUT, generator_lr, discriminator_lr)

