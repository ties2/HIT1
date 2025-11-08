Project Folder Structure
This README file is designed to guide you through the various components located in this project folder. It is not the main repository README, but rather a map to understand where each element of the proof of concept is stored.

Folder Overview
Code: Contains all scripts, datasets, and logs.
Poster: Stores the presentation poster in both PDF and PPTX formats.
Paper: Includes the final paper and related raw materials.
Presentation: Houses the presentation materials.

Code Folder
The main repository for the project is housed here. It includes:

README.md: Located at Code/README.md, this file details the project setup and usage instructions.

dataset: This subfolder contains various datasets used for training and testing the models:

Data Files:
test.npz: Used for testing the CNN1D, U-Net, and CNN3D models.
train_tiled.npz: Training data for the U-Net and CNN3D models.
train_patched.npz: Training data for the CNN1D model.
test_hh.npz, train_patched_hh.npz: HyperHue preprocessed versions of the test and train_patched datasets.
train_complete.npz, train_complete_hh.npz: Training datasets with all samples, used for training models on macro datasets.
test_macro_complete.npz, test_macro_complete_hh.npz: Test datasets including images taken with a macro lens.

CSV Files:
test.csv, train.csv, train_complete.csv, test_macro_complete.csv: These files list sample names and distribution details for the respective datasets.

Data and Macro Folders:
Contain .hsimage files, .json metadata, converted RGB images, and masks as .png files for non-macro and macro samples, respectively.

elements: Includes various pipeline elements.

log: Contains pretrained models with their specific training and testing datasets detailed:
cnn1d_best/model.npz: Best model, trained on train_patched.npz, tests on test.npz.
cnn1d_hh/model.npz: Trained on train_patched_hh.npz, tests on test_hh.npz.
cnn1d_macro/model.npz: Trained on train_complete.npz, tests on test_macro_complete.npz.
cnn1d_macro_hh/model.npz: Trained on train_complete_hh.npz, tests on test_macro_complete_hh.npz.
cnn3d_inference/model.npz: Inference model, loads CNN1D best automatically, tests on test.npz.
unet_best/model.npz: Best U-Net model, trained on train_tiled.npz, tests on test.npz.

hit_pipeline.py: Script for running model inference. Set to use log/cnn1d_best/model.npz on dataset/test.npz by default. To avoid overwriting pretrained models, change the model path when switching the training model.

Poster Folder
Contains both the PDF and PPTX files of the project poster.

Paper Folder
Milad_Isakhani_Zakaria_Paper.pdf: The final paper document.
Milad_Isakhani_Zakaria_Paper.zip: Archive of raw files used in the paper, including LaTeX files and images.

Presentation Folder
Houses the PowerPoint presentation file used for the final project presentation.