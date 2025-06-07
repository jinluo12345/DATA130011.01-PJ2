# CIFAR-10 CNN Experiments: VGG vs ResNet and Ablation Studies

## Overview

This repository contains the PyTorch implementation for the experiments described in the paper "Implementation of Neural Network for CIFAR-10 Image Classification". It provides code to train and evaluate various Convolutional Neural Network (CNN) architectures, including VGG variants and ResNet models, on the CIFAR-10 dataset. The project focuses on systematic ablation studies to understand the impact of different architectural components, hyperparameters, and training strategies, with a specific investigation into the effects of Batch Normalization on the loss landscape.

## Features

*   Implementation of multiple CNN architectures:
    *   VGG Variants: `VGG_A` (Base), `VGG_A_Light`, `VGG_A_Dropout`, `VGG_A_BN`, `VGG_A_Sigmoid`, `VGG_A_Tanh`, `VGG_Large`, `VGG_Res`, `VGG_Huge`
    *   ResNet Models: `ResNet-50`, `ResNet-101` (adapted for CIFAR-10)
*   Scripts for training models with configurable hyperparameters.
*   Code for running ablation studies on:
    *   Model Size
    *   Architecture (VGG vs ResNet vs VGG-Res)
    *   Activation Functions (ReLU, Sigmoid, Tanh)
    *   Dropout Regularization
    *   Batch Normalization
    *   Batch Size
    *   Optimizers (Adam, SGD variants)
    *   Learning Rate Schedulers (StepLR, CosineAnnealingLR)
    *   Data Augmentation
*   Visualization of training progress (loss and accuracy curves).
*   Script to generate the loss landscape comparison plot demonstrating the smoothing effect of Batch Normalization.
*   Standard CIFAR-10 data loading, preprocessing, and optional augmentation pipeline.

## Training

You'll likely have a main training script. You'll need to specify the model architecture and hyperparameters via config files

Example: Training the Base VGG Model (VGG_A)
```bash
python train.py --config=config/base.yaml
```
## Ablation

for further experiments, you can use different config files to do ablation experiments.
