import tensorflow as tf

def build_generator(SEED_SIZE, image_shape):

    filter_constant = 16
    l2reg = tf.keras.regularizers.l2()

    height = image_shape[0]
    width = image_shape[1]
    depth = image_shape[-1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(height*width*depth*filter_constant, input_shape=(SEED_SIZE,)))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((int(height/2), int(width/2), depth * 4 * filter_constant)))
    print(model.output_shape)

    model.add(tf.keras.layers.Conv2DTranspose(depth * filter_constant, (5, 5), strides=(2, 2), padding='same'))
    print(model.output_shape)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(depth * int(filter_constant/2), (5, 5), strides=(1, 1), padding='same'))
    print(model.output_shape)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(depth, (5, 5), strides=(1, 1), padding='same', activation='tanh'))
    print(model.output_shape)

    return model


def build_discriminator(img_shape, depth):

    const = ClipConstraint(0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(depth*64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape, kernel_constraint=const))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(depth*128, (5, 5), strides=(2, 2), padding='same', kernel_constraint=const))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
#
#
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)


def critic_loss(label, output):
    return tf.keras.backend.mean(label*output)


def wasserstein_generator_loss(fake_output):
    return -tf.keras.backend.mean(fake_output)


# clip model weights to a given hypercube
class ClipConstraint:
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}




