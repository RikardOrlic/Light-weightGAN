# Artistic Image Generation Using Generative Adversarial Networks

The focus is on generating pictures in the style of Gustav Klimt. 
The main problems were:
- the limited size of the available training dataset(consisting of 161 pictures that were made by Gustav Klimt)
- the limited computing power(single Nvidia GeForce GTX 1070)

## Light-weight GAN
Simple implementation of light-weight GAN structure proposed in <a href="https://openreview.net/forum?id=1Fqg133qRaI">'Towards Faster and Stabilized GAN Training for High-fidelity Few-Shot Image Synthesis'</a>, in PyTorch. 

<img src="./images/panda_gen_sample256.jpg"></img>

256x256 Pictures of pandas generated in about 16 hours of training on a single GPU.


<img src="./images/klimt_gen_sample256.jpg"></img>

256x256 Pictures of Klimt generated in about 16 hours of training on a single GPU. 

The panda pictures are of much better quality than the Klimt pictures, leading to the conclusion that the model isn't complex enough.


## Transfer learning with the Style-GAN3 model

Transfer learning resulted in generated pictures of better quality, but the training took over 11 days on the single GPU.

<img src="./images/klimt_gen_sample1024.jpg"></img>