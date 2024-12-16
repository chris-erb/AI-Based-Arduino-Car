# Road Line Detection Results

This repository demonstrates the performance and evaluation metrics of a trained road line detection model in YOLOv8. Below are the results including confusion matrices, evaluation curves, and loss metrics during training.

***Note: 'curved-road-line' detection was not implemented in the final implementation due to the poor depth of the ESP32***

---

## Dataset Examples

### Trained Road Line Samples
| **Image 1**            | **Image 2**            | **Image 3**            |
|-------------------------|------------------------|------------------------|
| ![Road 1](Flask_App/dataset/train/images/road3_jpg.rf.fb16f619d429374dfd2735eac210a287.jpg)   | ![Road 2](Flask_App/dataset/train/images/road1_jpg.rf.81250a6cc34ab45269761e455731e601.jpg)   | ![Road 3](Flask_App/dataset/train/images/road_and_curve2_jpg.rf.124f05553be229593c58e86249562ce3.jpg)   |

---

### Mask Samples (From Taped Lines)
| **Image 1**            | **Image 2**            | **Image 3**            |
|-------------------------|------------------------|------------------------|
| ![Road 1](Flask_App/dataset/masks/road1.jpg)   | ![Road 2](Flask_App/dataset/masks/road2.jpg)   | ![Road 3](Flask_App/dataset/masks/road3.jpg)   |

---

### Lane Detection ROI
![ROI](Flask_App/roi_mask.jpg)

---

## Evaluation Results

### Confusion Matrices
1. **Standard Confusion Matrix**  
   ![Confusion Matrix](Flask_App/runs/detect/train/confusion_matrix.png)

2. **Normalized Confusion Matrix**  
   ![Normalized Confusion Matrix](Flask_App/runs/detect/train/confusion_matrix_normalized.png)

---

### Performance Curves

1. **F1-Confidence Curve**  
   ![F1 Confidence Curve](Flask_App/runs/detect/train/F1_curve.png)

2. **Precision-Confidence Curve**  
   ![Precision Confidence Curve](Flask_App/runs/detect/train/P_curve.png)

3. **Precision-Recall Curve**  
   ![Precision Recall Curve](Flask_App/runs/detect/train/PR_curve.png)

4. **Recall-Confidence Curve**  
   ![Recall Confidence Curve](Flask_App/runs/detect/train/R_curve.png)

---

### Training and Validation Metrics

**Training Losses and Validation Metrics**  
![Results](Flask_App/runs/detect/train/results.png)

- **Box Loss** and **Classification Loss** for both training and validation.
- **Precision** and **Recall** metrics over epochs.
- mAP@50 and mAP@50-95 scores during training.
