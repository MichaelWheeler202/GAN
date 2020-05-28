# external libraries
from PIL import Image
from pathlib import Path
import time
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# local files
import networks


def import_data(FILE_PATH, resize_shape, filetypes):

    # Create array for all of our images
    count = 0
    first = None
    for ft in filetypes:
        for filename in Path(FILE_PATH).glob('**/*.' + ft):
            if count == 0:
                first = Image.open(filename).convert('RGB')

                if resize_shape is not None:
                    first = first.resize(resize_shape)

            count += 1

    arr_length = count

    first = np.asarray(first, dtype=np.uint8)

    height = first.shape[0]
    width = first.shape[1]
    depth = 0

    dimensions = first.ndim

    if dimensions == 3:
        depth = first.shape[2]

    image_list = np.zeros(shape=(arr_length, height, width, depth), dtype="float32")

    count = 0
    im = None
    for ft in filetypes:
        for filename in Path(FILE_PATH).glob('**/*.' + ft):

            im = Image.open(filename).convert('RGB')
            im.load()

            if resize_shape is not None:
                im = im.resize(size=resize_shape)

            data = np.asarray(im, dtype="float32")

            print(data.shape)

            image_list[count] = data/255
            count += 1

    if width != height:
        print("The width and height of the image are not the same.  Currently this is only designed to handle square images.")
        exit(1)

    if depth == 0:
        depth = 1
        image_list = image_list.reshape(len(image_list), height, width, depth)

    return width, height, depth, image_list


def generate_and_save_images(model, epoch, test_input, show, save_dir, save=False):

    predictions = model(test_input, training=False)
    predictions = tf.math.multiply(predictions, 255)

    for i in range(predictions.shape[0]):
        my_plot = plt.subplot(4, 4, i+1)
        if predictions.shape[-1] == 1:
            my_plot.imshow(predictions[i, :, :, 0], cmap='gray')
        else:
            my_plot.imshow(predictions[i, :, :, :])
        my_plot.axis('off')

    if show:
        plt.show()

    if save:
        plt.savefig(save_dir + '/' + 'image_at_epoch_{:04d}.png'.format(epoch))

@tf.function
def train_step(images, batch_size, seed_size, generator, discriminator,
               discriminator_ratio, generator_optimizer, discriminator_optimizer):

    noise = tf.random.normal([batch_size, seed_size])

    for i in range(discriminator_ratio):
        with tf.GradientTape() as disc_tape:

            real_output = discriminator(images, training=True)

            real_disc_loss = networks.critic_loss(label=-1, output=real_output)

        gradients_of_discriminator = disc_tape.gradient(real_disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            fake_output = discriminator(generated_images, training=True)

            fake_disc_loss = networks.critic_loss(label=1, output=fake_output)

        gradients_of_discriminator = disc_tape.gradient(fake_disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)


        gen_loss = networks.wasserstein_generator_loss(fake_output=fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def train(dataset, epochs, generator, discriminator,  generator_optimizer, discriminator_optimizer,
          seed, batch_size, seed_size, save_freq, discriminator_ratio, show_output, save_dir, save_output):

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, batch_size, seed_size, generator, discriminator,
                       discriminator_ratio, generator_optimizer, discriminator_optimizer)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)

        if (epoch)%save_freq == 0:
            generate_and_save_images(generator, epoch, seed, show_output, save_dir, save_output)

        print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))

        discriminator_fake_scores = 0
        discriminator_real_scores = 0

        for i in range(25):
            noise = tf.random.normal([1, seed_size])
            generated_image = generator(noise, training=False)
            decision = discriminator(generated_image, training=False)
            discriminator_fake_scores = discriminator_fake_scores + decision[0][0]
        # plt.imshow(generated_image[0,:,:,:])
        # plt.show()

        real_images = dataset.shuffle(60000).unbatch().batch(1)
        i = 0

        for e in real_images:

            if i >= 25:
                break

            decision = discriminator(e, training=False)
            discriminator_real_scores = discriminator_real_scores + decision[0][0]

            i += 1

        print('Generator Scores {}   Real Scores: {}'.format(discriminator_fake_scores/25, discriminator_real_scores/25))