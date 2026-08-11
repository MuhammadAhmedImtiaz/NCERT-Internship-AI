# Task 24 – Convolutional Neural Networks & Transfer Learning

## Part A & B: Custom CNN Architecture & Data Augmentation
- **Architecture:** Developed `CustomCIFARNet` featuring 3 Convolutional blocks ($3 \to 32 \to 64 \to 128$), Batch Normalization, Max Pooling, and Dropout ($p=0.3$).
- **Data Augmentation:** Implemented `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, and `ColorJitter()` to prevent overfitting on CIFAR-10.
- **Optimization & Scheduling:** Used `AdamW` paired with a `CosineAnnealingLR` scheduler over 10 epochs.

## Part C: Transfer Learning Benchmarking Matrix
Evaluated **ResNet18**, **VGG16**, and **MobileNetV2** across Feature Extraction and Fine-Tuning strategies:

- **Preprocessing Alignment:** Input images resized to $64 \times 64$ and normalized using ImageNet channel statistics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Fine-Tuning Safety:** Reduced learning rate ($10^{-4}$) during full backbone unfreezing to prevent destroying pretrained feature weights (catastrophic forgetting).

## Part D & E: Deployment Recommendations & Trade-Offs
1. **Cloud Deployment (No Latency Constraint):** **ResNet18 / ResNet50 Fine-Tuned** delivers the highest accuracy and macro F1-score due to residual skip connections.
2. **On-Device / Mobile Deployment:** **MobileNetV2 Feature Extractor** is recommended for low memory footprints and fast CPU/edge inference via depthwise separable convolutions.
3. **Artifact Persistence:** Model weights saved to `custom_cifar10_model.pth` and metrics stored in `transfer_learning_summary.csv`.