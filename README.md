# BranchDetection

Branch point detection in 3D neuron morphology reconstructions with low-resolution mouse brain images. This project implements a 3D U-Net with attention modules to detect branch and terminal points in automated neurite traces.

## 📜 Overview

- Trains and evaluates a deep learning model to correct topological errors in neurite traces.
- Uses 3D image stacks and labeled branch/terminal point masks.
- Outputs corrected branch point locations for post-processing in neural circuit reconstructions.

## 📁 File Structure

- `train.py` – standard training script using 3D U-Net + attention.
- `train_weighted.py` – weighted training with class imbalance handling.
- `train_generator.py` – data loader/generator for training batches.
- `U_attention.py` – attention-augmented 3D U-Net model.
- `U_test.py` – inference/prediction routines for branch points.
- `parameters.py` – hyperparameters and device setup.
- `ultis.py` – utility functions (e.g., evaluation, masks).
- `load_image.py` – image I/O and preprocessing.
- `transform_taichi.py` – data augmentation and transforms.
- `generate_augmented_images.py` – generates synthetic training variants.
- `Prediction_postprocess.ipynb` – post-processing and visualization notebook.
- `model_weights/` – pre-trained weights, training history, visual plots.

## 🏋️ Training

```bash
python train.py         # basic training
python train_weighted.py  # with class weights
```

## 🔍 Inference

```bash
python U_test.py
```

## 📦 Data

Expected input: 3D TIFF or MAT volumes with voxel annotations for branch and terminal points.

## 🛠️ GUI for Labeling

Manual annotation GUI described in the thesis can be found here:  
[https://github.com/venkatachalamlab/Segmentation_GUI](https://github.com/venkatachalamlab/Segmentation_GUI)

## 📚 Reference

This codebase supports the methods in Chapter 3 of the thesis:

> Deng, Hang. *Machine Learning for 3D Neuron Tracking and Identification in C. elegans, and Key Points Detection*. Northeastern University, 2025.
