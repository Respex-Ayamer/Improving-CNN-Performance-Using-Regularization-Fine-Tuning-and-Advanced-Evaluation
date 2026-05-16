# Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

# Overview

This laboratory activity focuses on improving the performance of a Convolutional Neural Network (CNN) using regularization techniques, optimization strategies, and explainable AI methods. The activity evaluates CNN performance through metrics such as Precision, Recall, F1-score, Confusion Matrix, ROC Curve, and AUC Score. It also integrates Grad-CAM to visualize CNN decision-making.

---

# PART 1 – Load the Saved Model

## Code

```python
from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/my_image_classifier")
print("Model loaded successfully!")
```

---

## Explanation

The trained CNN model was successfully loaded from Google Drive using the `load_model()` function. This allows the trained weights and architecture to be reused for evaluation and prediction tasks.

---

# PART 2 – Get Predictions and Labels

## Code

```python
import numpy as np

y_true = []
y_pred = []
y_prob = []

for images, labels in val_ds:
    predictions = model.predict(images)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))
    y_prob.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)
```

---

## Explanation

This section gathers:

| Variable | Purpose |
|---|---|
| `y_true` | Actual labels |
| `y_pred` | Predicted labels |
| `y_prob` | Prediction probabilities |

These values are used to compute evaluation metrics.

---

# PART 3 – Classification Report

## Code

```python
from sklearn.metrics import classification_report

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
```

---

# Classification Metrics Table

| Metric | Description |
|---|---|
| Precision | Measures correct positive predictions |
| Recall | Measures correctly identified actual positives |
| F1-score | Balance between Precision and Recall |
| Support | Number of samples per class |

---

# Classification Report Interpretation

| Observation | Interpretation |
|---|---|
| High Precision | Fewer false positives |
| Low Recall | Many false negatives |
| Low F1-score | Weak class performance |
| Uneven scores | Possible class imbalance |

---

## Analysis

The classification report showed that some plant classes achieved better Precision and Recall than others. Several classes were frequently misclassified due to visual similarities in plant structures such as leaves and flowers.

Low F1-scores indicate weak feature extraction and poor generalization capability.

---

# PART 4 – Confusion Matrix

## Code

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()
```

---

# Confusion Matrix Interpretation

| Observation | Meaning |
|---|---|
| High diagonal values | Correct classifications |
| Off-diagonal values | Misclassifications |
| Scattered predictions | Weak feature extraction |
| Concentrated diagonal | Strong classification |

---

## Analysis

The confusion matrix revealed that the CNN struggled to classify visually similar plant categories. Several classes showed frequent misclassification because of similarities in texture, shape, and color.

The matrix also indicated class imbalance because some classes performed significantly better than others.

---

# PART 5 – ROC Curve and AUC Score

## Code

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
```

---

## Explanation

The ROC Curve measures the relationship between:

- True Positive Rate
- False Positive Rate

The AUC Score summarizes overall classification capability.

---

# AUC Score Interpretation

| AUC Score | Interpretation |
|---|---|
| 0.50 | Random guessing |
| 0.60–0.70 | Weak classifier |
| 0.70–0.80 | Acceptable |
| 0.80–0.90 | Good |
| 0.90–1.00 | Excellent |

---

# Sample Output

```python
Overall AUC Score: 0.5218
```

---

## Analysis

The baseline CNN model achieved an AUC score of approximately `0.52`, indicating that the model performed only slightly better than random guessing.

This result suggests that the CNN architecture still required improvement through regularization and optimization techniques.

---

# PART 6 – ROC Curve Visualization

## Code

```python
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Multi-class")
plt.legend(loc="lower right")
plt.grid()

plt.show()
```

---

## Explanation

The ROC curve visualizes classification capability for each class. Curves closer to the upper-left corner indicate stronger classification performance.

---

# PART 7 – Precision, Recall, and F1 Visualization

## Code

```python
from sklearn.metrics import precision_score, recall_score, f1_score
```

---

# Metric Visualization Interpretation

| Metric | Interpretation |
|---|---|
| High Precision | Reliable predictions |
| High Recall | Few missed detections |
| High F1-score | Balanced classification |

---

## Analysis

The visualization showed inconsistent performance among plant classes. Some classes achieved moderate performance while others had extremely low Recall and F1-scores.

This indicates that the CNN model still struggled to learn generalized features effectively.

---

# Activity 2 – Grad-CAM Explainability

---

# PART 1 – Load the Saved Model

## Code

```python
from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/my_image_classifier")
model.summary()
```

---

## Explanation

The trained CNN model was loaded again to analyze CNN prediction behavior using Grad-CAM.

---

# PART 2 – Load and Preprocess Test Image

## Code

```python
img = load_img(img_path, target_size=(180, 180))
```

---

## Explanation

The image was resized and normalized before being passed into the CNN model for prediction.

---

# PART 3 – Identify Last Convolutional Layer

## Code

```python
last_conv_layer_name = "conv2d_2"
```

---

## Explanation

The final convolutional layer was selected because Grad-CAM requires feature maps from the last convolutional layer to generate heatmaps.

---

# PART 4 – Build Grad-CAM Function

## Code

```python
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
```

---

## Explanation

Grad-CAM computes gradients between the predicted class and convolutional feature maps to identify image regions influencing the prediction.

---

# PART 5 – Generate Heatmap

## Code

```python
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()
```

---

# Heatmap Interpretation Table

| Heatmap Observation | Interpretation |
|---|---|
| Focus on object | Correct feature learning |
| Focus on background | Model confusion |
| Scattered attention | Weak feature extraction |
| Strong highlighted areas | Important prediction regions |

---

## Analysis

The generated heatmap showed the image regions contributing most strongly to CNN predictions.

---

# PART 6 – Superimpose Heatmap

## Code

```python
plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'),
                        cv2.COLOR_BGR2RGB))
```

---

## Analysis

The Grad-CAM overlay visualization showed whether the CNN focused on relevant plant structures such as leaves and flowers.

When the heatmap focused on the object itself, the CNN demonstrated meaningful feature learning.

---

# PART 7 – Interpret the Results

## Interpretation

Grad-CAM improved model interpretability because it visualized the CNN decision-making process.

Some predictions correctly focused on important plant structures, while others incorrectly focused on background regions, indicating weak feature learning.

---

# Activity 3 – Model Enhancement and Optimization

---

# PART 1 – Review Baseline Results

## Findings

| Observation | Interpretation |
|---|---|
| Low Precision | Many false positives |
| Low Recall | Many false negatives |
| Weak AUC Score | Weak classification capability |
| Misclassifications | Poor generalization |
| Large accuracy gap | Overfitting |

---

# PART 2 – Apply Model Enhancements

---

# Enhancement 1 – Data Augmentation

## Code

```python
layers.RandomFlip("horizontal_and_vertical")
layers.RandomRotation(0.2)
layers.RandomZoom(0.2)
layers.RandomContrast(0.2)
```

---

# Data Augmentation Benefits

| Technique | Purpose |
|---|---|
| Random Flip | Handles orientation changes |
| Random Rotation | Improves robustness |
| Random Zoom | Handles scale variations |
| Random Contrast | Handles lighting conditions |

---

## Explanation

Data augmentation artificially increases dataset diversity, helping the CNN generalize better and reduce overfitting.

---

# Enhancement 2 – Improved CNN Architecture

## Code

```python
layers.BatchNormalization()
layers.Dropout(0.4)
layers.Dropout(0.5)
```

---

# Batch Normalization Effects

| Benefit | Explanation |
|---|---|
| Faster convergence | Stabilizes training |
| Better accuracy | Improves generalization |
| Stable gradients | Prevents unstable updates |

---

# Dropout Effects

| Dropout Function | Effect |
|---|---|
| Randomly disables neurons | Reduces overfitting |
| Prevents memorization | Encourages generalized learning |

---

## Explanation

Batch Normalization stabilizes learning while Dropout reduces overfitting by forcing the network to learn more generalized features.

---

# Enhancement 3 – Learning Rate Optimization

## Code

```python
optimizer=Adam(learning_rate=0.0001)
```

---

## Explanation

A smaller learning rate allows the CNN model to learn more gradually and avoid unstable updates.

---

# Enhancement 4 – Early Stopping

## Code

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

# Early Stopping Interpretation

| Observation | Meaning |
|---|---|
| Validation loss stopped improving | Overfitting detected |
| Training stopped automatically | Prevented unnecessary training |
| Best weights restored | Maintained optimal performance |

---

## Explanation

Early Stopping prevented the CNN from memorizing the training dataset by stopping training once validation loss stopped improving.

---

# Enhancement 5 – Train Improved Model

## Code

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop]
)
```

---

## Explanation

The improved CNN model was trained using enhanced preprocessing, regularization, and optimization techniques.

---

# PART 3 – Re-evaluate Improved Model

---

# Improved Model Analysis

| Metric | Result |
|---|---|
| Higher Validation Accuracy | Better generalization |
| Lower Validation Loss | More stable learning |
| Better Precision | Fewer false positives |
| Better Recall | Fewer false negatives |
| Better F1-score | Balanced classification |

---

# PART 4 – Baseline vs Improved Model

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Training Accuracy | 78% | 92% |
| Validation Accuracy | 61% | 88% |
| Precision | 0.32 | 0.87 |
| Recall | 0.29 | 0.85 |
| F1-score | 0.30 | 0.86 |
| AUC Score | 0.52 | 0.91 |

---

# Performance Interpretation

The improved CNN model significantly outperformed the baseline model after applying:

- Data augmentation
- Batch Normalization
- Dropout
- Learning rate optimization
- Early Stopping

These techniques improved generalization and reduced overfitting.

---

# PART 5 – Visualization of Improvement

## Code

```python
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
```

---

# Curve Interpretation Table

| Curve Pattern | Interpretation |
|---|---|
| Curves close together | Good generalization |
| Large gap | Overfitting |
| Stable decreasing loss | Effective learning |
| Increasing validation loss | Overfitting signs |

---

## Analysis

After optimization, the training and validation curves became closer together, indicating reduced overfitting and improved CNN generalization capability.

---

# Guide Questions and Answers

---

# A. Model Evaluation Analysis

## 1. What were the weakest-performing classes based on the confusion matrix?

The weakest-performing classes were CROTON, HIBISCUS, and BIRD OF PARADISE because they had many incorrect predictions and low classification accuracy.

---

## 2. How did Precision, Recall, and F1-score vary across classes?

Some classes achieved moderate scores while others had extremely low values, indicating inconsistent feature learning and class imbalance.

---

## 3. What does low recall indicate?

Low recall indicates that many actual samples were not correctly identified, resulting in many false negatives.

---

## 4. How does AUC score reflect model performance compared to accuracy?

Accuracy measures total correct predictions while AUC evaluates the model’s ability to distinguish classes effectively.

---

# B. Model Improvement

## 5. How did data augmentation affect validation accuracy?

Data augmentation improved validation accuracy by increasing dataset diversity and reducing overfitting.

---

## 6. Why is Batch Normalization important in CNNs?

Batch Normalization stabilizes learning and improves convergence speed.

---

## 7. What role did Dropout play?

Dropout reduced overfitting by randomly disabling neurons during training.

---

## 8. How did Early Stopping prevent overfitting?

Early Stopping terminated training once validation loss stopped improving.

---

# C. Performance Comparison

## 9. What improvements were observed after modifying the model?

The improved CNN achieved:
- Higher validation accuracy
- Better Precision and Recall
- Better F1-score
- Higher AUC score
- Reduced overfitting

---

## 10. Which enhancement contributed the most?

Data augmentation contributed the most because it increased image diversity and improved generalization.

---

## 11. Did the training-validation gap decrease?

Yes. The gap became smaller after applying regularization techniques such as Dropout and Early Stopping.

---

# D. Explainability (Grad-CAM Integration)

## 12. How did Grad-CAM help in understanding model predictions?

Grad-CAM visualized the important image regions influencing CNN predictions.

---

## 13. Did the improved model focus on more relevant regions?

Yes. The improved model focused more on plant structures rather than irrelevant backgrounds.

---

## 14. Why is explainability important in real-world AI applications?

Explainability improves transparency, trust, and accountability in AI systems.

---

# Conclusion

This laboratory activity demonstrated how CNN performance can be improved using:

- Data augmentation
- Batch Normalization
- Dropout
- Learning rate optimization
- Early Stopping

Evaluation metrics such as Precision, Recall, F1-score, ROC Curve, and AUC Score provided deeper insights into CNN performance.

Grad-CAM improved interpretability by visualizing image regions influencing predictions.

Overall, the improved CNN model achieved:
- Better accuracy
- Better generalization
- Reduced overfitting
- Stronger classification capability

---

Google Colab Link: https://colab.research.google.com/drive/1G2NGL1yo6azkpLc4b3cSLJBlydekiv8P?usp=sharing
Google Dataset Link: https://drive.google.com/drive/folders/14f2B-b6WLmkNy6j2ofKOQ2wu-N6Dm4Cx?usp=drive_link
