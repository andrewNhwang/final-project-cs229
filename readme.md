# Overview
This model uses a conditional GAN to generate fake segmentations of a hand after being fed a real RGB image of a hand. The generator uses a U-net encoder-decoder structure with skip connections to create a segmentation mask that should bypass bottleneck of information. The discriminator, conditioned on the real RGB image determines the likelihood that patches of scale n x n are real/fake and then takes an average of all patches to determine the likelihood that this image is a real/fake segmentation of the real RGB (I think this could be improved: tried to implement https://arxiv.org/abs/1904.00284 but kept on getting mode collapse. Github is not yet published but coming out soon). Basic blocks are of form Convolution - BatchNorm - ReLU. geoGANModel is additionally trained with an additional loss term meant to constrain the segmentation mask to be constant to common geometric transforms such as rotation or reflection but the generator is the same -should provide with stronger results. The variational autoencoder resembles largely the same structure but with using a resnet encoder-decoder. Trained with 2 main distinguishing datasets: one with only one hand images (15027) and one with one hand images (15027) and multi-hand images (4000 - this was harder to come by). Best parameters were with a vanilla gan loss, patch size of 50 x 50, and a normal gaussian weight initialization (tried with lsgan/wganp, 1x1/128*128/256x256, and kaiming weight initalization). Needs to be trained on lots of data - prone to overfitting. Could also be used just for synthetic data generation.

# Training

To combine datasets, go into cGAN/datasets and then run:
python pair.py --fold_A testingSamples/A --fold_B testingSamples/B --fold_AB testingSamples/

Import parameters)
--dataroot datasets/testingSamples (or whatever folder you have images located in)
--model cGAN (or geoGAN)
--name <insert random word>
--gpu_ids 0 (-1 for cpu)

Example
python train.py --dataroot ./datasets/testingSamples/ --gpu_ids -1  --model cGAN --name random


# Testing
Import parameters)
--dataroot datasets/testingSamples/A (specific directory of images)
--model test
--name testing (insert checkpoint name)
--gpu_ids 0 (-1 for cpu)

Example:
python test.py --model test --name testing --dataroot datasets/testingSamples/A --gpu_ids 0


Additional Parameters can be found in options baseOptions.
