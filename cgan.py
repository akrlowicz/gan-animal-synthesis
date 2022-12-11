from matplotlib import pyplot as plt
from keras.layers import InputLayer, Dense, Dropout
from keras.layers import LeakyReLU, Reshape, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy
from keras.metrics import Mean

import tensorflow as tf
from keras.utils import to_categorical

# define cGAN

# Structure based on code from https://keras.io/examples/generative/conditional_gan/

class ConditionalGAN(Model):
    def __init__(self, num_classes, img_rows=80, img_cols=80, latent_dim=100):
        super(ConditionalGAN, self).__init__()

        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        self.generator_in_channels = self.latent_dim + self.num_classes
        self.discriminator_in_channels = self.channels + self.num_classes

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self):
        super(ConditionalGAN, self).compile()
        self.d_optimizer =  Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer =  Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        # Adam(lr=0.0003)
        self.loss_fn = BinaryCrossentropy(from_logits=True)

    def build_discriminator(self):

        model = Sequential(name="discriminator")

        model.add(InputLayer((self.img_rows, self.img_cols, self.discriminator_in_channels)))
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5 ,5), strides=(2 ,2), padding='same')) 	# downsample to 40x40
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5 ,5), strides=(2 ,2), padding='same')) # downsample to 20x30
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5 ,5), strides=(2 ,2), padding='same')) # downsample to 10x10
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5 ,5), strides=(2 ,2), padding='same')) # downsample to 5x5
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1))

        model.summary()


        return model


    def build_generator(self):


        model = Sequential(name = 'generator')

        model.add(InputLayer((self.generator_in_channels,)))
        model.add(Dense(5 * 5 * self.generator_in_channels))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((5, 5, self.generator_in_channels)))

        model.add(Conv2DTranspose(128, (4 ,4), strides=(2 ,2), padding='same')) # upsample to 10x10
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4 ,4), strides=(2 ,2), padding='same')) # upsample to 20x20
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4 ,4), strides=(2 ,2), padding='same')) # upsample to 40x40
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4 ,4), strides=(2 ,2), padding='same')) # upsample to 80x80
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, (5 ,5), activation='tanh', padding='same')) # output layer 80x80x3

        model.summary()


        return model


    def train_step(self, data):

        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        # add dummy dimensions to the labels so that they can be concatenated with the images (for the discriminator instead of embedding)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=[self.img_rows * self.img_cols])
        image_one_hot_labels = tf.reshape(image_one_hot_labels, (-1, self.img_rows, self.img_cols, self.num_classes))

        # sample random points in the latent space and concatenate the labels for the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)


        generated_images = self.generator(random_vector_labels)

        # combine generated images with real images
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)

        # assemble labels discriminating real from fake images.
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)

        # assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))


        # train generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)


        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


class GANMonitor(Callback):
    def __init__(self, num_classes, epoch_summarize=5, n=7, latent_dim=100):
        self.epoch_summarize = epoch_summarize
        self.n = n
        self.latent_dim = latent_dim
        self.num_classes = num_classes


    def on_epoch_end(self, epoch, logs=None):

        if (epoch +1) % self.epoch_summarize == 0:

            random_latent_vectors = tf.random.normal(shape=(self.n * self.num_classes, self.latent_dim))
            labels = tf.repeat(tf.reshape(tf.range(0, self.num_classes), (-1 ,1)), self.n)
            one_hot_labels = to_categorical(labels)
            random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)

            generated_images = self.model.generator(random_vector_labels)
            generated_images = (generated_images + 1) / 2.0 # scale from [-1,1] to [0,1]

            for i in range(self.n * self.num_classes):
                plt.subplot(self.n, self.num_classes, 1 + i)
                plt.axis('off')
                plt.imshow(generated_images[i])

            # save plot to file
            filename = 'generated_plot_e%03d.png' % (epoch +1)
            plt.savefig(filename)
            plt.close()


            filename = 'generator_model_%03d.h5' % (epoch +1)
            self.model.generator.save(filename)