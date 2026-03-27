### Improve simple-image-clssification

## 🚀 Improvements

Compared to the baseline CNN, the following improvements were applied:

- Data augmentation (random crop, horizontal flip)
- Batch normalization for stable training
- Dropout to reduce overfitting
- Increased training epochs (10 → 20)

These changes significantly improved generalization performance.

| Model | Accuracy |
|------|----------|
| Baseline SimpleCNN | 74.68% |
| ImprovedCNN | **81.40%** |

---

## 📈 Analysis

- Training becomes more stable with batch normalization  
- Data augmentation reduces overfitting  
- Dropout improves generalization  

## 📊 Training Visualization

### Loss Curve
![Loss Curve](outputs/loss_curve.png)

### Accuracy Curve
![Accuracy Curve](outputs/accuracy_curve.png)

The improved model continues to learn effectively beyond 10 epochs,
achieving peak performance at epoch 19.

---

## 📌 Key Improvement

+6.72% accuracy gain over baseline