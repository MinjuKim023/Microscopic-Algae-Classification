# Microscopic Algae Classification: Toward Enhancing Water Quality

This repository presents the source code and documentation for a deep learning-based algae classification project conducted during the Spring 2024 semester as part of a Computer Vision course in the MS in Data Science program at Indiana University Bloomington.

## Project Overview

Water quality concerns in Monroe Lake, the primary water source for Bloomington, Indiana, have increased due to algae blooms during warmer months. Traditional algae monitoring tools such as FlowCam often miss up to 90% of images due to strict filtering criteria. This project proposes a deep learning approach to automate the classification of algae species using image-based methods, contributing toward better public health outcomes and environmental monitoring.

This work was completed in collaboration with the City of Bloomington, supported by data and mentorship from Jill Minor, and supervised through coursework led by Professor David Crandall.

## Objectives

- Classify microscopic images of algae into 12 distinct categories, including species such as *Anabaena*, *Aphanazomenon*, *Cyclotella*, *Planktothrix*, and *Black Hole* artifacts.
- Compare the performance of several deep learning models (CNN, ResNet, U-Net, MobileNet).
- Improve performance on morphologically ambiguous taxa like *Straight Anabaena*.
- Mitigate class imbalance and lens artifact issues using image augmentation and curated preprocessing.

## Methodology

### Dataset

- The dataset included 1,287 manually verified and annotated images derived from a larger set of 3,060 records provided by the City of Bloomington.
- Label refinement resulted in 12 target classes, including one additional class ("Black Hole") for lens artifacts.
- Images were standardized to 224x224 pixels using custom padding to avoid distortion.

**Note**: Due to privacy and policy constraints, the dataset used in this project is not included in this public repository.

### Preprocessing

- Dataset extraction from a SQLite3 database.
- Mapping and merging of species labels into biologically meaningful categories.
- Balanced data via augmentation: rotation, flipping, distortion, and zoom, applied only to training/validation sets.
- Mean and standard deviation calculated using `get_mean_std.py` to normalize images.

### Models

Four deep learning architectures were implemented and evaluated:

| Model      | Strengths |
|------------|-----------|
| CNN        | Lightweight, interpretable baseline |
| ResNet     | High classification accuracy with residual connections |
| U-Net      | Excellent reconstruction performance for biomedical-like images |
| MobileNet  | Mobile-friendly architecture with reduced parameters |

Each model accepts an input of shape `(3, 224, 224)` and produces a `(batch_size, 12)` output tensor.

### Evaluation

The models were evaluated using the following metrics:
- Accuracy
- Loss
- True Positive Rate (TPR)
- False Positive Rate (FPR)

Particular focus was placed on *Straight Anabaena*, which is visually similar to other species and often misclassified.

#### Results Summary

| Model      | Accuracy | Loss   | TPR (S. Anabaena) | FPR (S. Anabaena) |
|------------|----------|--------|-------------------|-------------------|
| CNN        | 83.33%   | 0.666  | 0.48              | 0.20              |
| ResNet     | 94.93%   | 0.131  | 0.96              | 0.17              |
| U-Net      | 94.20%   | 0.091  | 0.92              | 0.17              |
| MobileNet  | 76.09%   | 0.573  | 0.44              | 0.42              |

ResNet demonstrated the highest accuracy and robustness, particularly in distinguishing *Anabaena* subtypes.

## Repository Contents

| File                | Description |
|---------------------|-------------|
| `main.py`           | CLI-based training loop for model selection and evaluation |
| `model.py`          | Implementation of CNN, ResNet, UNet, and MobileNet |
| `dataset_class.py`  | PyTorch dataset loader and transformer |
| `data_augmentation.py` | Custom dataset builder, augmentor, and label organizer |
| `get_mean_std.py`   | Script to compute normalization parameters |
| `utils.py`          | Contains early stopping logic and other helper utilities |
| `Algae_poster.pdf`  | Final poster summarizing the research |
| `Algae_final paper.pdf` | Final written report submitted for course completion |

## Deployment & Usage

To train a model:

```
python main.py --epochs 100 --model_class 'ResNet' --batch_size 128 --learning_rate 0.001 --l2_regularization 0.0
```

### CLI Options

- `--model_class` (str): Choose among `CNN`, `ResNet`, `UNet`, `MobileNet`
- `--epochs` (int): Number of training epochs
- `--batch_size` (int): Mini-batch size
- `--learning_rate` (float): Optimizer learning rate
- `--l2_regularization` (float): Regularization weight decay

## Limitations and Future Work

- Due to class similarity, distinguishing straight-shaped algae (e.g., *Straight Anabaena*) remains challenging.
- Further improvements could be achieved by adding more labeled data and exploring transformer-based architectures.
- Deployment on real-time streaming data for automated field monitoring is a promising next step.
- The current dataset lacks metadata such as magnification or sample collection time, which may enhance model reliability.

## Contribution Statement

This project was a team collaboration. The repository has been made public by Minju Kim for the purpose of sharing the research outcomes and code implementation.

Minju Kim's contributions:
- Led the end-to-end data preprocessing and label mapping process.
- Developed the data augmentation pipeline and managed class balancing.
- Served as the point of contact with the City of Bloomington, translating expert feedback into actionable model design.
- Authored the final report, poster content, and model interpretability analysis.
- Created most of the project visuals and analysis figures.

Two other team members contributed significantly to model development, database handling, and training pipeline engineering. Their efforts are acknowledged, and this repository does not redistribute their names or institutional accounts for privacy reasons.

## Acknowledgements

Special thanks to Jill Minor, Data Analyst at the City of Bloomington, for providing domain expertise and algae image data. This project would not have been possible without her support.

Appreciation also goes to Professor David Crandall for fostering the collaboration and providing academic guidance throughout the semester.

---

Contact: Minju Kim  
Email: mk159@iu.edu  
Website: [Insert website link here]
