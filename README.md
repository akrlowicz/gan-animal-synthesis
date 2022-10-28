# What's that (GAN)imal - generating new animal images using GANs
Author: Alicja Karlowicz, 12127661, TU Wien

## Idea and motivation
Since a while I wanted to play around with GANs and I reckon this is a perfect opportunity to do so. While my initial idea of generating food recipes failed at the state of planning, I turned to different problems. GANs are known to work well with images. The most popular examples of their use are generating fake faces or style-transfer (e.g. generating Mona Lisa in Monet style). I thought of something different to generate that could be equally as fun and unpredictable in the outcome - generating animal images. I'm interested if the images would be of high quality to even tell that this can be an animal and if GAN will produce a picture of e.g. random cat or will it manage to fuse some pictures of different animals to generate "new species". Latter would be my desired goal, as I consider it more interesting. Therefore my plan is to feed a dataset of different animals e.g. mammals, birds, fish and train my GAN to synthesize a new one.

I acknowledge that getting meaningful outcome would be challenging. Images can be of low-resolution of just pixels creating some shapes. However, for me it is an opportunity to learn about GANs in their image-2-image form and play around with them, before moving in the future to other exciting projects like image-2-text generation.

## Related work
GANs have many applications and many variants suitable for different tasks The big application of GANs is image synthesis. The following survey provides valuable insights on GANs type for this task, approaches, loss functions and evaluation methods. [[1]](#1)

When we take a look at https://thisxdoesnotexist.com/, it seems like GANs can generate almost anything. On website we can find images of faces, cats, memes, food, rental places, landscapes generated artificially. When it comes to animals, Ta-Yieng Chen in a blog post described using DCGAN in attempt to generate animal images from dataset of tigers, lions, dogs, foxes and cats. [[2]](#2) After 100 epochs the results are quite patchy, nevertheless a pixelated shape of animal face can be seen in synthetic photo.

Far more developed work was published by MIT Media Lab as "Meet the ganimals" project. [[3]](#3) The researchers used BigGAN to generate novel, hybrid animals and released them to the public to react to their aesthetic. We can find a hybrid of golden retriver and goldfish or pug mixed with starfish.


## Dataset 
Few datasets that can be tested in this experiment:
- **Animal Faces-HQ dataset (AFHQ)** - 15.000 images of 512x512 resolution of cats, dogs and wildlife [[4]](#4)
	- + dataset is of high quality, with portraits of animals, which are centered and focused on their faces
	- - quite small size
- **AnimalWeb dataset** - 22.400 images of animal faces from 350 diverse species and 21 animal orders [[5]](#5)
	- + larger diversity
	- - high variations in pose, scale and backgrounds
-  **Animals-10 Kaggle dataset** - 28.000 images of animals from google of 10 classes (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant) [[6]](#6)
	- + less classes
	- - medium quality, not only focused on faces
- subset of **ImageNet** - 396 classes of animals [[7]](#7)
	- + this was used for MIT Media Lab project, probably huge dataste, can sample desired animals
	- - effort put in subsampling and picking categories, majority of classes are dog breeds -> lack of relative diversity, unsure how many images combined

## Setup
Project will be implemented in **Python 3.x** using Google Colab GPU resources.

## Project plan
In total I plan to spend ~95h on hacking phase of this task and ~33h on building the application and summarizing results. That results in total of ~128h of work. At this point it's hard to estimate time realistically, so I'm curious how accurate this breakdown will be.

More theoretical knowledge gathering: **5h**
- reading papers on GANs for image-2-image generation

Dataset preparation: **15h**
- getting datasets
- preprocessing
- combining datasets
- cleaning

GAN preparation: **25h**
- setting up environment
- using chosen GAN implementation
- tweaking it to my task

Training iterations: **40h**
- evaluation
- fine-tuning

Analysis of results: **10h**
- qualitative and quantitative analysis
- producing representative results

Application building: **25h**
- implementation of simple web page
- deployment

Final presenting: **8h**
- writing report
- designing presentation


## References
<a id="1">[1]</a>  *Image Synthesis with Adversarial Networks: a Comprehensive Survey and Case Studies*, Shamsolmoali et al.  https://arxiv.org/pdf/2012.13736.pdf

<a id="2">[2]</a> *Create New Animals using DCGAN with PyTorch*,  Ta-Ying Cheng https://towardsdatascience.com/create-new-animals-using-dcgan-with-pytorch-2ce47810ebd4

<a id="3">[3]</a> *Interpolating GANs to Scaffold Autotelic Creativity*, Epstein et al. http://ceur-ws.org/Vol-2827/CAC-Paper_3.pdf

<a id="4">[4]</a> Animal faces HQ dataset https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq

<a id="5">[5]</a> *AnimalWeb: A Large-Scale Hierarchical Dataset of Annotated Animal Faces*, Khan M. H. et al. https://salman-h-khan.github.io/papers/CVPR20_AnimalWeb.pdf

<a id="6">[6]</a> Animals-10 dataset https://www.kaggle.com/datasets/alessiocorrado99/animals10

<a id="7">[7]</a> ImageNet https://www.image-net.org/index.php
