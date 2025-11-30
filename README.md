# Image Recoloring With GAN Architecture

Purpose of this task is to recolor images from the cityscapes dataset.


I choose GAN with UNet as the generator and simple downsampling encoder for discriminator.
At first I chose just UNet with MSE loss, but the training got stuck 
after a while at a very low loss (~0.00002) and didn't produce colorful images, mainly 
struggling with highly saturated parts.

That's why I chose GAN to try and produce more realistically looking images (and also because I just
like the architecture). Main inspiration was the following paper: 
https://link.springer.com/chapter/10.1007/978-3-319-94544-6_9.

the GAN implementation produced reasonable results.


GAN Architecture:
![Architecture](https://github.com/Depresivna-ryza/image-recoloring-gan/blob/master/final_files/final_model_architecture.png?raw=true)
