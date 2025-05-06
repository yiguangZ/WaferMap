# 🧪 Wafer Map Classification with CNNs and VGG16 (ECE 157A/272A)

This project implements and compares custom Convolutional Neural Networks (CNNs) and transfer learning using the VGG16 model to classify wafer map images. These maps represent manufacturing defects in semiconductor wafers and are categorized into specific failure patterns.

## 🎯 Objective

Train a deep learning model to automatically classify wafer map images into multiple failure pattern classes using both a custom CNN and a fine-tuned VGG16 architecture.

## 🧰 Features

- 📊 Loads and processes real-world wafer map defect dataset
- 🧠 Implements a custom PyTorch CNN with convolutional, pooling, dropout, and linear layers
- 🏗 Fine-tunes a pre-trained VGG16 model for transfer learning
- 🧪 Evaluates model accuracy and loss
- 📉 Visualizes training/validation performance

## 📂 File Structure

- `ECE_157A_272A_F24_Homework3_wafermap_nn_undergraduate.ipynb`: Main notebook with full code
- `data/`: Folder containing the wafer map dataset (to be downloaded separately)
- `README.md`: Project overview and setup

## 🖥️ Libraries Used

- `torch`, `torchvision`
- `matplotlib`, `numpy`, `sklearn`
- `PIL`, `os`, `glob`

## 🚀 How to Run

1. Download and unzip the wafer map dataset from the UCSB Box link (provided in class).
2. Open and run the notebook in Jupyter or Google Colab.
3. The notebook includes all necessary functions for:
   - Dataset loading
   - Model definition
   - Training loop
   - Evaluation and prediction

## ✍️ Author

Yiguang Zhu (based on UCSB ECE 157A/272A Fall 2024 coursework)
