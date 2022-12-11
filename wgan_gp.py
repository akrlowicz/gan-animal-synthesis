from matplotlib import pyplot as plt
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, LayerNormalization
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.initializers import RandomNormal

# define WGAN

# Large amount of credits go to: https://keras.io/examples/generative/wgan_gp/
# which the structure of my code was based on. Architecture is custom.


class WGAN(Model):
    def __init__(self, d_steps=5, img_rows=80, img_cols=80, latent_dim=100):
        super(WGAN, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = 10.0

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()


    def compile(self):
        super(WGAN, self).compile()
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # the loss functions for the generator (note that its negative)
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)


    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp



    def build_generator(self):
        n_nodes = 512 * 5 * 5

        init = RandomNormal(stddev=0.02)

        model = Sequential(name="generator")

        model.add(Dense(n_nodes, input_dim=self.latent_dim, kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((5, 5, 512)))
        model.add(UpSampling2D())

        model.add(Conv2D(256, kernel_size=(4, 4), padding="same", kernel_initializer=init))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=(4, 4), padding="same", kernel_initializer=init))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=(4, 4), padding="same", kernel_initializer=init))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        model.add(Conv2D(self.channels, kernel_size=(5, 5), padding="same", kernel_initializer=init))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        init = RandomNormal(stddev=0.02)
        model = Sequential(name="discriminator")

        model.add(Conv2D(64, kernel_size=(5 ,5), strides=(2 ,2), input_shape=self.img_shape, padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(5 ,5), strides=(2 ,2), padding="same", kernel_initializer=init)) # downsample to 40x40
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=(5 ,5), strides=(2 ,2), padding="same", kernel_initializer=init)) # downsample to 20x20
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=(5 ,5), strides=(2 ,2), padding="same", kernel_initializer=init)) # downsample to 10x10
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.


        # train discriminator
        for i in range(self.d_steps):

            random_latent_vectors = tf.random.normal \
                (shape=(batch_size, self.latent_dim)) # get points from latent vector

            with tf.GradientTape() as tape:

                fake_images = self.generator(random_latent_vectors, training=True) # decode to fake images

                fake_logits = self.discriminator(fake_images, training=True) # fake images
                real_logits = self.discriminator(real_images, training=True) # real images

                d_cost = self.discriminator_loss(real_img=real_logits, fake_img=fake_logits) # using the fake and real image logits get discriminator loss

                # gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight # add gradient penalty to original loss

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))


        # train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True) # get fake imgs

            gen_img_logits = self.discriminator(generated_images, training=True) # discriminator logits for fake images
            g_loss = self.generator_loss(gen_img_logits)

        # get the gradients with respect to the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(Callback):
    def __init__(self, epoch_summarize=5, n=10, latent_dim=100):
        self.epoch_summarize = epoch_summarize
        self.n = n
        self.latent_dim = latent_dim


    def on_epoch_end(self, epoch, logs=None):

        if (epoch +1) % self.epoch_summarize == 0:

            random_latent_vectors = tf.random.normal(shape=(self.n * self.n, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images + 1) / 2.0 # scale from [-1,1] to [0,1]

            for i in range(self.n * self.n):
                plt.subplot(self.n, self.n, 1 + i)
                plt.axis('off')
                plt.imshow(generated_images[i])

            # save plot to file
            filename = 'generated_plot2_e%03d.png' % (epoch +1)
            plt.savefig(filename)
            plt.close()


            filename = 'generator_model2_%03d.h5' % (epoch +1)
            self.model.generator.save(filename)