# Enhanced Tuberculosis Bacilli Detection using Attention-Residual U-Net, Vision Transformer, and Ensemble Classification

This repository contains code for tuberculosis bacilli detection using an Attention-Residual U-Net for image segmentation, followed by two classification approaches: a Vision Transformer (ViT) and an ensemble classifier (SVM, Random Forest, XGBoost, and Voting Classifier).

---

## Repository Structure

```plaintext id="pc5cf8"
/
├── segmentation/         # Contains the segmentation code using Attention-Residual U-Net
├── classification/       # Contains the ensemble classification code (SVM, RF, XGB, and Voting Classifier)
├── utils/                # Helper scripts for data preprocessing and evaluation
├── dataset/              # Full microscopic image dataset and corresponding ground truth
├── ViT.py                # Vision Transformer architecture for RoI classification
└── README.md             # Instructions and details about the project
```

---

## Requirements

To run the code, ensure you have the following libraries installed:

* Python 3.8+
* TensorFlow 2.x
* NumPy
* OpenCV
* Scikit-learn
* XGBoost

You can install the required packages using:

```bash id="jkv3od"
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Segmentation

Navigate to the segmentation/ folder.

Run the segmentation script to generate masks for input images:

```bash id="tztccq"
python unet_segmentation.py
```


### 2. Vision Transformer Classification

The `ViT.py` file contains the Vision Transformer architecture for classifying Regions of Interest (RoIs).

You can import it as:

```python id="6xl7wp"
from ViT import create_vit_model

model = create_vit_model()
```

---

## Preprocessing and Utilities

Helper functions for data preprocessing and evaluation are provided in the `utils/` folder.

---

## Dataset

This study utilizes **full-field sputum smear microscopic images** acquired under laboratory conditions using Ziehl–Neelsen (ZN) staining.

Each raw image is paired with a corresponding **ground truth image**:

* Ground truth images contain **expert annotations** in the form of visual overlays
* **Circles** indicate isolated bacilli
* **Rectangles/squares** indicate bacilli clusters

### Data Organization

The complete dataset included in this repository contains **101 microscopic field images** and their corresponding annotated ground truth images:

```plaintext id="ov2k0p"
dataset/
├── images/
└── ground_truth/
```

* Ground truth images share the **same filenames** as their corresponding raw images
* Each image–ground truth pair represents one annotated microscopic field

### Image Format

* Images are provided in **TIFF format (.tif)** to preserve fine-grained microscopic details
* This avoids compression artifacts and maintains diagnostic quality

### Availability

* The **complete dataset** associated with this work is publicly available via Zenodo:
  **https://doi.org/10.5281/zenodo.XXXXXXX**

* The repository contains both the **full dataset** and the **implementation code** required for reproducibility

---

## Acknowledgment Requirement

Users of this dataset are requested to appropriately acknowledge the State TB Cell, Government of Kerala, India, and the Government District TB Hospital, Ernakulam, Kerala, India (under NTEP), for providing the sputum smear samples and ground truth annotations.

---

## Citation


Vision Transformer architecture implementation for classifying RoIs into bacilli and non bacilli.

1. This code is related to the article titled as "Tuberculosis Bacilli Detection Enhanced: A Hybrid Attention Residual U-Net and Vision Transformer Approach" which is under consideration for publication at "The Visual Computer", Springer journal.  

---

## Dataset Citation

If you use this dataset, please cite both dataset and the article submitted at The Visual Computer as:

1. Dataset -
```bibtex id="ow8f5r"
@dataset{greeshma_tb_dataset_2026,
  author       = {Greeshma K and Vishnukumar S},
  title        = {Tuberculosis Bacilli Detection Dataset: Ziehl–Neelsen Stained Sputum Smear Microscopic Images with Expert Annotations},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19280029},
  url          = {https://doi.org/10.5281/zenodo.19280029}
}

```
2. Article is under consideration at "The Visual Computer", Springer journal. 

---
