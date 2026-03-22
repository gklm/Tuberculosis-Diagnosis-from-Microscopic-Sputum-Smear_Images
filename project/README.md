Enhanced Tuberculosis Bacilli Detection using Attention-Residual U-Net and Ensemble Classification

This repository contains code for tuberculosis bacilli detection using an Attention-Residual U-Net for image segmentation and an ensemble classifier (SVM, Random Forest, XGBoost, and Voting Classifier) for classification.

Repository Structure

/
├── segmentation/         # Contains the segmentation code using Attention-Residual U-Net
├── classification/       # Contains the ensemble classification code (SVM, RF, XGB, and Voting Classifier)
├── utils/                # Helper scripts for data preprocessing and evaluation
└── README.md             # Instructions and details about the project

Requirements

To run the code, ensure you have the following libraries installed:

Python 3.8+

TensorFlow 2.x

NumPy

OpenCV

Scikit-learn

XGBoost

You can install the required packages using:

pip install -r requirements.txt

Usage Instructions

1. Segmentation

Navigate to the segmentation/ folder.

Run the segmentation script to generate masks for input images.

python unet_segmentation.py

2. Classification

Navigate to the classification/ folder.

Ensure your dataset is organized into folders for different classes.

Run the ensemble classification script to train the models.

python ensemble_classifier.py

3. Preprocessing and Utilities

Helper functions for data preprocessing and evaluation are provided in the utils/ folder.

Citation


@article{greeshma2025automated,
  title={Automated tuberculosis diagnosis: A hybrid approach using attention-residual U-Net segmentation with ensemble classification},
  author={Greeshma, K and Vishnukumar, S},
  journal={Franklin Open},
  pages={100479},
  year={2025},
  publisher={Elsevier}
}


Vision Transformer architecture implementation for classifying RoIs into bacilli and non bacilli.
@misc{k2025efficientaccuratetuberculosisdiagnosis,
      title={Efficient and Accurate Tuberculosis Diagnosis: Attention Residual U-Net and Vision Transformer Based Detection Framework}, 
      author={Greeshma K and Vishnukumar S},
      year={2025},
      eprint={2501.03538},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.03538}, 
}

Dataset

This study utilizes full-field sputum smear microscopic images acquired under laboratory conditions.

Each raw image is paired with a corresponding ground truth image:

Ground truth images contain expert annotations in the form of visual overlays
Circles indicate isolated bacilli
Rectangles/squares indicate bacilli clusters
Data Organization

A small subset of the dataset is included for demonstration:

sample_data/
├── images/
├── ground_truth/
Ground truth images share the same filenames as their corresponding raw images
Each image–ground truth pair represents one annotated microscopic field
Image Format
Images are provided in TIFF format (.tif) to preserve fine-grained microscopic details
This avoids compression artifacts and maintains diagnostic quality
Availability
The dataset included in this repository is a limited sample for demonstration purposes
The complete dataset will be made publicly available upon acceptance/publication

Users of this dataset are requested to appropriately acknowledge the State TB Cell, Government of Kerala, India, and the Government District TB Hospital, Ernakulam, Kerala, India (under NTEP), for providing the sputum smear samples and ground truth annotations.
