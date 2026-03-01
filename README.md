# Conclusion

## Summary

I have successfully developed and evaluated multiple deep learning models for tomato leaf disease detection across **10 distinct disease categories**. The primary objective was to create a lightweight, accurate model suitable for deployment on mobile devices.

## Models Developed

### 1. Custom CNN Model
A custom convolutional neural network built from scratch with the following architecture:
- **4 convolutional blocks** with progressive channel expansion (32 → 64 → 128 → 256)
- **Global Average Pooling** for dimensionality reduction
- **Fully connected layers** with dropout regularization
- **Key advantage**: Lightweight and fast inference, optimized for mobile deployment
- **Training strategy**: Used weighted cross-entropy loss to handle class imbalance, with early stopping based on validation F1-weighted score

### 2. EfficientNet-B0 (Full Fine-Tuning)
Advanced transfer learning with all layers trainable:
- Started with ImageNet pre-trained weights as initialization
- Unfroze classifier layer and do feature extraction with higher learning rate for 5 epochs
- Unfroze all layers for end-to-end training
- Used best hyperparameters found during hyperparameter search to preserve pre-trained knowledge and enable fine specialization
- Custom classification head adapted for 10 disease classes
- **Key advantage**: Superior accuracy through domain-specific adaptation and transfer learning

## Model Performance Comparison

### Evaluation Metrics
All models were evaluated on the test set using:
- **Loss**: Cross-entropy loss on test samples
- **Accuracy**: Overall percentage of correct predictions
- **F1-Weighted**: Weighted average F1 score accounting for class imbalance
- **F1-Macro**: Unweighted average F1 score across all disease classes
- **Confusion Matrix**: Per-class performance visualization

### Test Set Results

| Model | Loss | Accuracy | F1-Weighted | F1-Macro |
|-------|------|----------|-------------|----------|
| Custom CNN | 0.130 | 95.75 % | 95.77 % | 95.63 % |
| EfficientNet-B0 (Full Fine-Tuning) | 0.085 | 98.23 % | 98.23 % | 98.23 % |

*Note: Actual metrics are logged in MLflow and displayed in the evaluate_model() cell outputs above.*

### Key Observations
- **Accuracy vs. Efficiency trade-off**: Custom CNN provides reasonable accuracy with minimal computational cost, while EfficientNet-B0 achieves superior accuracy at higher computational expense
- **Transfer learning advantage**: EfficientNet benefits from pre-trained ImageNet features, enabling better generalization
- **Class imbalance handling**: Weighted loss and class weights helped both models learn minority disease classes effectively
- **Confusion matrices**: Reveal which disease pairs are frequently confused, informing potential improvements

## Training Improvements Applied

- **Class weights**: Inverse frequency weighting to mitigate dataset imbalance
- **Data augmentation**: Leveraged augmented training samples to improve generalization
- **Early stopping**: Prevented overfitting by monitoring validation F1-weighted score
- **Learning rate scheduling**: Adaptive learning rate adjustment
- **Hyperparameter search**: Optimal hyperparameter search using results yielded by Optuna
- **Regularization**: Dropout layers and weight decay to reduce overfitting
