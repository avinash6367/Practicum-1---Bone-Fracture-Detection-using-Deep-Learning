# Bone Fracture Detection using Deep Learning

## Project Overview
This project aims to develop an AI-based system for detecting bone fractures from X-ray images. Early and accurate detection is crucial in healthcare, but manual diagnosis can be time-consuming and prone to human error. By applying deep learning and computer vision techniques, our goal is to build a reliable decision-support tool that can assist radiologists and improve patient outcomes.

## Objectives
- Use Convolutional Neural Networks (CNNs) and transfer learning models (e.g., ResNet, EfficientNet) to classify X-ray images as fracture vs. no fracture.  
- Perform data preprocessing (normalization, augmentation, denoising) to improve model robustness.  
- Evaluate model performance using accuracy, precision, recall, F1-score, and AUC.  
- Apply Grad-CAM and similar methods to make the model interpretable.  

## Datasets
We are using two publicly available datasets:  
1. [Bone Fracture Detection Computer Vision Dataset (ResearchGate)](https://www.researchgate.net/publication/382268240_Bone_Fracture_Detection_Computer_Vision_Project)  
2. [MURA: Musculoskeletal Radiographs Dataset (Nature Scientific Data)](https://www.nature.com/articles/s41597-023-02432-4)  

The datasets contain thousands of X-ray images with fracture and non-fracture labels. Estimated combined size: 20–30 GB.  

## Tech Stack
- Python  
- TensorFlow / PyTorch  
- OpenCV for image preprocessing  
- Matplotlib / Seaborn for visualization  
- scikit-learn for evaluation metrics  

## Planned Timeline (8 Weeks)
- Week 1: Literature review, dataset exploration, and repository setup  
- Week 2: Data preprocessing and augmentation  
- Week 3: Build baseline CNN model  
- Week 4: Implement transfer learning models  
- Week 5: Model training and hyperparameter tuning  
- Week 6: Evaluation and visualization (ROC curves, Grad-CAM)  
- Week 7: Model comparison and optimization  
- Week 8: Final report and presentation  

## Anticipated Challenges
- Class imbalance (more non-fracture than fracture images) → will use augmentation and class-weighted loss.  
- High computational requirements → mitigate with transfer learning and cloud GPU resources.  
- Model interpretability → Grad-CAM heatmaps to highlight regions influencing predictions.  

## Repository Structure (initial draft)
