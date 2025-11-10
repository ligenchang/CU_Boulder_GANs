# Photo to Monet: CycleGAN Style Transfer

This is PyTorch implementation of CycleGAN for transforming photographs into Monet-style paintings, built for the Kaggle competition "I'm Something of a Painter Myself."

## Overview

This project uses Generative Adversarial Networks (GANs) to learn Monet's artistic style—his color palette, brushstrokes, and impressionist feel—and apply it to regular photos. The challenge is that we don't have paired examples (photo + matching Monet painting), so we can't use standard supervised learning approaches.

Instead, the model learns to map between two unpaired domains: real photographs and Monet paintings. It figures out what makes a painting look "Monet-ish" and transfers those characteristics to new photos.

## Architecture

The implementation uses a **lightweight CycleGAN** with two generators and two discriminators:

- **Generator G_AB**: Transforms photos into Monet-style paintings
- **Generator G_BA**: Transforms Monet paintings back to photo-realistic images
- **Discriminator D_A**: Evaluates whether Monet images are real or generated
- **Discriminator D_B**: Evaluates whether photos are real or generated

Each generator uses an encoder-decoder architecture with 6 residual blocks (instead of the typical 9 for faster training), reflection padding, and instance normalization. The discriminators use PatchGAN architecture with 4 convolutional layers to evaluate local image patches.

**Total parameters**: ~17 million

## Dataset

- **Monet paintings**: 300 images
- **Real photos**: 7,038 images
- All images preprocessed to 256×256 pixels

## Training Details

### Loss Functions
The model combines three loss components:

1. **Adversarial Loss** (BCEWithLogitsLoss): Makes generated images look realistic
2. **Cycle Consistency Loss** (L1): Ensures photo → Monet → photo reconstruction preserves content (weight: 10)
3. **Identity Loss** (L1): Maintains color composition (weight: 0.5)

### Hyperparameters
- Epochs: 4
- Batch size: 4
- Learning rate: 0.0003 with cosine annealing
- Optimizer: Adam (β1=0.5, β2=0.999)
- Weight decay: 1e-4
- Label smoothing: 0.9 for discriminator stability

## Results

The model showed stable convergence over 4 epochs:

- Generator loss: 3.65 → 3.16
- Discriminator A (Monet) loss: ~0.45
- Discriminator B (Photo) loss: ~0.59

The generated images demonstrate softer, painterly qualities with warmer color palettes and subtle brushstroke textures. However, the transformations are relatively subtle due to limited training epochs.