# Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer (ACAI)
# Reimplementation of the paper https://arxiv.org/abs/1807.07543

**Authors**: Grigorii Sotnikov, Vladimir Gogoryan, Dmitry Smorchkov and Ivan Vovk (all have equal contribution)

The work has been done as the Deep Learning Course final project "*Controllable Face Generation via Conditional Latent Models*" at Skoltech. You can check the report and details in `demo/paper.pdf`.

## Info

Autoencoders provide a powerful framework for learning compressed representations by encoding all of the information needed to reconstruct a data point in a latent code. In some cases, autoencoders can “interpolate”: By decoding the convex combination of the latent codes for two datapoints, the autoencoder can produce an output which semantically mixes characteristics from the datapoints. In
this paper, we propose a regularization procedure which encourages interpolated outputs to appear more realistic by fooling a critic network which has been trained to recover the mixing coefficient from interpolated data. We then develop a simple benchmark task where we can quantitatively measure the extent to which various autoencoders can interpolate and show that our regularizer dramatically improves interpolation in this setting. We also demonstrate empirically that our regularizer produces latent codes which are more effective on downstream tasks, suggesting a possible link between interpolation abilities and learning useful representations.

## Results

Our results of interpolation between latent codes of images sized to `64x64` on `CelebA` by `ACAI` trained with combination of `MSE` and `Perceptual Loss`.

![alt text](https://github.com/ivanvovk/controllable-face-generation/blob/ACAI/results/acai_interpolation.png)

### Inference and training

The training code for `CelebA` dataset can be found in `ACAI_CelebA.ipynb`. It uses `.py` files.
