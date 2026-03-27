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

## ⭐Inference
![test](https://github.com/user-attachments/assets/7677c5f6-f90e-4dcb-8592-18d4c53c62d8)
Prediction: cat

Confidence: 0.8645

<img width="996" height="525" alt="test_dog" src="https://github.com/user-attachments/assets/51f97b4a-e6f2-4b2e-bb6a-77274e8094a6" />
Prediction: dog

Confidence: 0.8596

![airplane](https://github.com/user-attachments/assets/41049a06-6c7f-4518-8653-5796c03c6976)
Prediction: airplane

Confidence: 0.9998

![automobile](https://github.com/user-attachments/assets/c5e4509a-7607-455e-a776-5ff01da4e6c6)
Prediction: automobile

Confidence: 1.0000

![bird](https://github.com/user-attachments/assets/2ad5b0a2-8264-477e-920c-7aa8f7f81edc)
Prediction: bird

Confidence: 0.8870

![deer](https://github.com/user-attachments/assets/5c4c0549-10fb-42b7-bd54-ad0e5ea2dfc6)
Prediction: deer

Confidence: 0.5277

![frog](https://github.com/user-attachments/assets/ce2b9964-3a4a-4a1f-98a0-575716428a14)
Prediction: bird

Confidence: 0.3217

![horse](https://github.com/user-attachments/assets/a69650af-b2a3-4422-bc8a-4a91323a451e)
Prediction: horse

Confidence: 0.9995

![ship](https://github.com/user-attachments/assets/4ef83115-9907-400a-acaf-95d0579ce338)
Prediction: ship

Confidence: 1.0000

![truck](https://github.com/user-attachments/assets/0e386773-bf7e-4926-bfdf-67e700884b6b)
Prediction: truck

Confidence: 0.9153






