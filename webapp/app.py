import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import subprocess

# PAGE CONFIG
st.set_page_config(page_title='GANimals', page_icon=':tiger:', layout="centered")

# GLOBAL VARS
CWD = os.getcwd()
LATENT_DIM = 100
CONDITIONAL = False
ONEHOT = False
NUM_CLASSES = 0

# UTIL FUNCTIONS
@st.cache(allow_output_mutation=True)
# @st.experimental_singleton
def load(model_name):

    # cond_gan_afd : generator_model_200.h5
    # cond_wgan_afd : generator_model_200.h5
    # dcgan_afd :generator_model _150
    # wgan_afd : generator_model2_200.h5
    # dcgan_afhq : generator_model_100.h5

    path = f"{CWD}/ganimals/{model_name}/generator_model_200.h5"

    if model_name == 'wgan_afd': path = path.replace('model', 'model2')
    elif model_name == 'dcgan_afd' : path = path.replace('200', '150')
    elif model_name == 'dcgan_afhq': path = path.replace('200', '100')

    # alternative way of accessing the models from github url
    # if not os.path.isfile(model_name + "_model.h5"):
    #     url_to_model = f'https://github.com/akrlowicz/gan-animal-synthesis/blob/main/ganimals/{path}?raw=true'
    #     st.write(url_to_model)
    #     urllib.request.urlretrieve(url_to_model, model_name + "_model.h5")


    generator = load_model(path)

    return generator

@st.cache
def load_classes(path_to_npz):
    data = np.load(path_to_npz, allow_pickle=True)
    classes = data['arr_0'].item()

    return classes

# @st.cache(show_spinner=False)
def generate_image(generator, n=5, latent_dim=100, conditional=False, onehot_encoded=True):

    if conditional:
        interpolation_noise = tf.random.normal(shape=(NUM_CLASSES, latent_dim))
        classes = tf.cast(list(classes_dict.values()), tf.uint8)

        if onehot_encoded:
            onehot_classes = to_categorical(classes)
            noise_and_labels = tf.concat([interpolation_noise, onehot_classes], 1)
            fake_img = generator.predict(noise_and_labels)
        else:
            fake_img = generator.predict([interpolation_noise, classes])
    else:
        # sample noise for the interpolation.
        interpolation_noise = tf.random.normal(shape=(n, latent_dim))
        fake_img = generator.predict(interpolation_noise)

    return fake_img

# @st.cache(show_spinner=False)
def plot_generated(examples):
    examples = (examples + 1) / 2.0

    fig = plt.figure(figsize=(17, 17))
    for i in range(len(examples)):
        plt.subplot(1, len(examples), 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])

    return fig

@st.cache(show_spinner=False)
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = [(1.0 - ratio) * p1 + ratio * p2 for ratio in ratios]  # uniform interpolation between two points in latent space

    return np.asarray(vectors)


@st.cache(show_spinner=False)
def interpolate_class(generator, first_number, second_number, n_steps=10, latent_dim=LATENT_DIM, onehot_encoded=True):
    # sample noise for the interpolation.
    interpolation_noise = tf.random.normal(shape=(n_steps, latent_dim))

    # calculate the interpolation vector between the two labels
    if onehot_encoded:
        # convert the start and end labels to one-hot encoded vectors
        first_number = to_categorical([first_number], NUM_CLASSES)
        second_number = to_categorical([second_number], NUM_CLASSES)

        first_number = tf.cast(first_number, tf.float32)
        second_number = tf.cast(second_number, tf.float32)

    percent_second_label = tf.linspace(0, 1, n_steps)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = ((1 - percent_second_label) * first_number + second_number * percent_second_label)

    # using fact that for cGAN I used one-hot encoding but for cWGAN label encoding with embedding, therefore input is different
    if onehot_encoded:
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = generator.predict(noise_and_labels)
    else:
        fake = generator.predict([interpolation_noise, interpolation_labels])

    return fake

def display_interpolated(generator, num_interpolation, n_steps=10, yA=None, yB=None, onehot_encoded=True):
    for i in range(0, num_interpolation):

        if CONDITIONAL:
            fake_images = interpolate_class(generator, yA, yB, n_steps, LATENT_DIM, onehot_encoded)
        else:
            interpolation_noise = tf.random.normal(shape=(2, 100))
            interpolated = interpolate_points(interpolation_noise[0], interpolation_noise[1], n_steps)
            fake_images = model.predict(interpolated)

        st.write(plot_generated(fake_images))

def model_name_format(model_name, dataset_name):
    return model_name.replace('c', 'cond_').replace('-GP', '').lower() + '_' + dataset_name.lower()


# ***************   LOADING/INIT   ***************
st.title("What's that GANimal? :dog: :cat: :tiger: :cow:")
st.write('Demnostration of hybrid animal faces synthesis via Generative Adverserial Networks')

st.sidebar.title("Menu")
st.sidebar.write("****")

# ***************   DATASET PICK   ***************

dataset_name = st.sidebar.radio('Choose the dataset', options=['AFD', 'AFHQ'])

if dataset_name == 'AFHQ':
    model_options = ['DCGAN']
else:
    model_options = ['DCGAN','WGAN-GP', 'cGAN', 'cWGAN-GP']

st.sidebar.write("Please note that current version only supports DCGAN for AFHQ dataset.")
st.sidebar.write("****")


# ***************   MODEL PICK   ***************
model_name = st.sidebar.selectbox('Choose the model', model_options)
model_name = model_name_format(model_name, dataset_name)
model = load(model_name)

st.sidebar.write("If the choice of the model is DCGAN or WGAN-GP, the synthesized images by model's generator are displayed.\n "
         "In the case of choosing cGAN or cWGAN-GP interpolation of images between generated images of particular class is displayed.")



# determine if we use conditional models and if the class is one hot encoded
CONDITIONAL = model_name.split('_')[0] == 'cond'
ONEHOT = model_name.split('_')[1] == 'gan'


if not CONDITIONAL:
    # ***************   GENERATION   ***************
    st.subheader(':point_right: Generation of sample images on classic model')

    n = st.slider('Number of pictures', 1, 10, 5, disabled=CONDITIONAL)


    with st.spinner('Wait for it...'):
        images = generate_image(model, n, conditional=CONDITIONAL, onehot_encoded=ONEHOT)
        fig = plot_generated(images)

    st.write(fig)

else:
    # load classes dict regardless (for displaying in drop down select box)
    classes_dict = load_classes(CWD + '/ganimals/afd_class_dict.npz')
    classes_list = [str(values) + ') ' + keys for keys, values in zip(classes_dict.keys(), classes_dict.values())]
    NUM_CLASSES = len(classes_list)

    # ***************   INTERPOLATION   ***************
    st.subheader(':point_right: Interpolation of images on conditional model')

    left_column, _, right_column = st.columns([1,0.2,2])
    n_steps = left_column.slider('Number of transitions', 1, 10, 5)
    num_interpolation = left_column.slider('Number of interpolations', 1, 10, 3)

    # class picker selectbox - disable if we dont use conditional models
    categoryA = right_column.selectbox('Choose first class', classes_list, 2)
    categoryB = right_column.selectbox('Choose second class', classes_list, 5)

    yA = int(categoryA.split(')')[0])
    yB = int(categoryB.split(')')[0])

    with st.spinner('Wait for it...'):
        display_interpolated(model, num_interpolation=num_interpolation, n_steps=n_steps, yA=yA, yB=yB, onehot_encoded=ONEHOT)

st.write("author: Alicja Karlowicz")

# disable menu on upper right corner
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)