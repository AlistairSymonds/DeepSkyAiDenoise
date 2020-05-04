This is my research/prrof of concept code into denoising astronomical images using deep learning, 
specifically with an aim to be used in making 'pretty pictures' for astrophotography.

Normally denoising training is done by taking an image, adding syntehtic noise and then using the 
original as the reference for the network. 

Instead here we're training based off of how multiple exposures are averaged to create one integrated image with reduced noise.

To that end there is a dataset processor, it expects the following structure - where for each channel the subs and integrated image are all aligned.
```
dataset_top
\-target/gear identifier
    |-channel_1.fits
    |-channel_1
    |    |-sub1.fits
    |    |-sub2.fits
    |    |-sub3.fits
    |    \-sub4.fits
    |
    |-channel_2.fits
    \-channel_2
         |-sub1.fits
         |-sub2.fits
         \-sub3.fits

```

Then running preprocess.py will chop that up into a collection of patches for the same 
coordinates across each sub



#
Needs:
* pytorch
* torchvision
* kornia
* numpy
* astropy