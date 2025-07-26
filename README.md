# Image Classification of Cats, Dogs, and Pandas using Fine-Tuned VGG19 CNN Model

## Problem Statement

The aim of this project is to build an image classification system that can accurately identify **cats** and  **dogs** using deep learning. By fine-tuning the pretrained **VGG19** model on a labeled dataset of animal images, this project demonstrates the effectiveness of **transfer learning** in solving multi-class classification tasks with high accuracy and minimal training time.



## Dataset Overview

- **Classes**: 2 — Cat, Dog  
- **Image Type**: Colored (RGB)  
- **Image Size (after preprocessing)**: 224 × 224 pixels  

### Preprocessing Steps

- Resized all images to 224×224 (required for VGG19 input)
- Normalized using ImageNet mean and standard deviation
- (Optional) Data augmentations like horizontal flip, rotation, etc.

This structured and preprocessed dataset enables effective training of the VGG19 model for multi-class classification.



##  VGG19 Model and Its Working

VGG19 is a deep Convolutional Neural Network (CNN) developed by the Visual Geometry Group (VGG) at the University of Oxford. Known for its simplicity and use of small 3×3 filters, VGG19 is highly effective when fine-tuned on smaller custom datasets.

###  Architecture Overview

- **Total Layers**: 19  
  - 16 convolutional layers  
  - 3 fully connected layers  
- **Convolutional Layers**:
  - 3×3 filters with stride=1 and padding=1
  - ReLU activation after each conv layer  
  - MaxPooling (2×2) after conv blocks  
- **Fully Connected Layers**:
  - Two FC layers with 4096 neurons  
  - Final FC layer replaced with 3-class output (cat, dog, panda)

###  How It Works

1. **Input**: An RGB image resized to 224×224
2. **Feature Extraction**: Passed through multiple conv+pool layers that extract features like edges, fur, and shape
3. **Flattening**: Output is flattened before entering FC layers
4. **Classification**: Final FC layer with softmax outputs probability scores for 3 classes
5. **Prediction**: The class with the highest score is selected

### Transfer Learning

- The pretrained weights from ImageNet are reused
- Only the final layer is modified and trained on our dataset
- Reduces training time and improves accuracy


##  Training Details

- **Epochs**: 3  
- **Batch Size**: 2400 (training), 600 (testing)  
- **Loss Function**: CrossEntropyLoss  
- **Optimizers**: Adam / SGD  
- **Training Time Monitoring**: `time.time()` used to log duration

At each step:
- Image batches were passed through the VGG19 model
- Predictions were compared with true labels
- Loss was computed and backpropagated
- Weights were updated


## Evaluation and Results

The model was trained over **3 epochs** with accuracy and loss trends recorded batch-wise.

### Epoch 1:
- Started: ~66.67% accuracy  
- Ended: 94.43% accuracy  
- Significant loss reduction (strong learning curve)

### Epoch 2:
- Started: ~90.48%  
- Ended: 97.17%  
- Most batches showed 0.0000 loss  
- Minor spikes (e.g., batch 181: loss = 7.9210) had no effect

### Epoch 3:
- Started: 94.05%  
- Peaked at: 97.85%  
- Few loss spikes (e.g., batch 361: 4.83, batch 401: 14.31)  
- High overall stability

### Accuracy Trend (Epoch-Wise):

| Epoch | Starting Accuracy | Final Accuracy |
|-------|-------------------|----------------|
| 1     | 66.67%            | 94.43%         |
| 2     | 90.48%            | 97.17%         |
| 3     | 94.05%            | 97.75%         |

The model quickly generalized well, with strong results achieved by just the second epoch.



## Key Libraries Used

- `torch`, `torchvision` – for model and training  
- `matplotlib` – for visualization  
- `os`, `PIL`, `numpy` – for file handling and image preprocessing  

## Prediction
<img width="539" height="542" alt="image" src="https://github.com/user-attachments/assets/ab4388e4-8b11-4449-bdd4-95fa3005b953" />
dog

## Confusion Matrix (for batch-size=20)
<img width="706" height="530" alt="image" src="https://github.com/user-attachments/assets/73a0a17a-c483-4a21-8266-52069de7ce3a" />



## Future Improvements

- Add a real-time prediction interface (e.g., Gradio)  
- Use learning rate schedulers for better training control  
- Unfreeze more convolutional layers for fine-tuning deeper representations



