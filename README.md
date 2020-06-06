# Controllable Face Generation via Conditional Adversarial Latent Autoencoder (ALAE)

**Authors**: Grigorii Sotnikov, Vladimir Gogoryan, Dmitry Smorchkov and Ivan Vovk (all have equal contribution)

The work has been done as the Deep Learning Course final project "*Controllable Face Generation via Conditional Latent Models*" at Skoltech. You can check the report and details in `demo/paper.pdf`.

## Info

![alt-text-0](demo/progressive-strategy.gif "progressive-strategy")

We implemented our own ALAE model, but stopped training at 64x64 resolution due to the resources-consuming process.

## Reproducibility

### Inference and training

Training code is available in `train_alae.ipynb` notebook.

### Data

The model is trained on CelebA128 dataset.

## Latent attributes manipulation experiments

Switch the branches of the repository for another experiments. We implemented facial keypoints transfer model, which manipulates just the latent space of ALAE, and tried to solve the "Neural Talking Heads" task. Also instead of facial keypoints we tried to condition the latent space mapping network with other CelebA class attributes. This experiments were done with ACAI-based (https://arxiv.org/pdf/1807.07543.pdf) encoder-decoder network.
