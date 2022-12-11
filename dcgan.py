from matplotlib import pyplot as plt

import tensorflow as tf

from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import Mean
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Flatten, LeakyReLU, Dropout
from keras.layers import Dense, Conv2D, Conv2DTranspose


# define DCGAN

# Large amount of credits go to: https://keras.io/examples/generative/dcgan_overriding_train_step/
# which the structure of my code was based on. Architecture is custom.

class DCGAN(Model):
    def __init__(self, img_rows = 80, img_cols = 80, latent_dim=100):
        super(DCGAN, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = latent_dim

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

    def compile(self):
        super(DCGAN, self).compile()

        self.g_optimizer = Adam(lr=0.0003)
        self.d_optimizer = Adam(lr=0.0003)

        self.loss_fn = BinaryCrossentropy()
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def build_discriminator(self):
        model = Sequential(name="discriminator")
        model.add(Conv2D(128, (5, 5), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))  # downsample to 40x40
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))  # downsample to 20x30
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))  # downsample to 10x10
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))  # downsample to 5x5
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):
        n_nodes = 128 * 5 * 5

        model = Sequential(name='generator')

        # foundation for 5x5 feature maps
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((5, 5, 128)))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # upsample to 10x10
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # upsample to 20x20
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # upsample to 40x40
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # upsample to 80x80
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, (5, 5), activation='tanh', padding='same'))  # output layer 80x80x3

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))  # get samples from latent spacr
        generated_images = self.generator(random_latent_vectors)  # decode to fake images
        combined_images = tf.concat([generated_images, real_images], axis=0)  # combine with real ones for discriminator

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
                           axis=0)  # create labels for images (1 real, 0 fake)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # adding random noise to the labels is important

        # train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))  # Assemble labels that say "all real images"

        # train the generator without updating weights of discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


class GANMonitor(Callback):

    def __init__(self, epoch_summarize=10, n=10, latent_dim=100):
        self.epoch_summarize = epoch_summarize
        self.n = n
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.epoch_summarize == 0:

            random_latent_vectors = tf.random.normal(shape=(self.n * self.n, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images + 1) / 2.0  # scale from [-1,1] to [0,1]

            for i in range(self.n * self.n):
                plt.subplot(self.n, self.n, 1 + i)
                plt.axis('off')
                plt.imshow(generated_images[i])

            # save plot to file
            filename = 'generated_plot_e%03d.png' % (epoch + 1)
            plt.savefig(filename)
            plt.close()

            filename = 'generator_model_%03d.h5' % (epoch + 1)
            self.model.generator.save(filename)