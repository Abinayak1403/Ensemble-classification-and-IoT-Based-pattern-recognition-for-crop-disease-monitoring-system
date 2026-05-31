# Ensemble-classification-and-IoT-Based-pattern-recognition-for-crop-disease-monitoring-system

#  Corn Leaf Disease Detection using Ensemble Learning
Download the datset from Kaggle :https://www.kaggle.com/datasets/atuls1124/corn-all-dataset

##  Project Overview

This mini-project aims to develop an intelligent pattern recognition system using **Ensemble Learning** for detecting diseases in **corn leaves**. The project uses image data to train a machine learning model capable of classifying leaf images into various disease categories. The approach combines feature extraction, preprocessing, and a Bagging ensemble model with SVM to enhance classification accuracy.

---

##  Objectives

- Build a robust classification model for corn leaf disease detection.
- Apply **ensemble learning** using BaggingClassifier with **Support Vector Machine (SVM)** as the base estimator.
- Evaluate the model using standard metrics: Accuracy, Precision, Recall, and F1-Score.
- Visualize the dataset and classification results for better understanding.

---

##  Project Structure

```
Corn_Leaf_Disease_Detection/
│
├── PR_Mini_Project.ipynb        # Jupyter Notebook with complete code
├── README.md                    # Project overview and documentation
├── Corn_Leaf_Disease_Dataset/  # Dataset directory (after extraction)
│   ├── blight/
│   ├── rust/
│   └── healthy/
```

---

##  Technologies Used

- **Python** 🐍
- **OpenCV** - Image processing
- **NumPy** - Numerical computation
- **scikit-learn** - ML models and evaluation
- **Matplotlib** - Visualization

---

##  Dataset Information

The dataset consists of images of corn leaves categorized into different folders:
- `blight`: Corn leaves affected by blight.
- `rust`: Corn leaves affected by rust.
- `healthy`: Healthy corn leaves.

**Note**: The dataset is expected to be uploaded as a `.zip` file which will be extracted inside the notebook.

---

## 🔧 Setup Instructions (Google Colab)

1. Open the notebook in Google Colab.
2. Upload the dataset ZIP file when prompted:
```python
from google.colab import files
uploaded = files.upload()
```
3. The notebook extracts the uploaded dataset:
```python
import zipfile

for filename in uploaded.keys():
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("/content/Corn_Leaf_Disease_Dataset")
```

4. Proceed with training and evaluation.

---

##  Model Overview

###  Ensemble Method: Bagging Classifier

- Reduces variance and avoids overfitting.
- Trains multiple instances of base learners and aggregates predictions.

###  Base Estimator: Support Vector Classifier (SVC)

- A powerful classification algorithm suitable for high-dimensional image data.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

model = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=42)
```

---

##  Preprocessing Steps

- Image resizing
- Grayscale conversion
- Feature flattening
- Label encoding using `LabelEncoder`
- Train-Test Split (typically 80-20 split)

---

##  Evaluation Metrics

- **Accuracy Score**
- **Classification Report**: Includes Precision, Recall, and F1-Score for each class
- **Confusion Matrix** (optional addition)

Sample output:
```plaintext
              precision    recall  f1-score   support

     blight       0.92      0.91      0.91        50
     healthy      0.95      0.96      0.95        50
     rust         0.94      0.92      0.93        50

    accuracy                           0.93       150
```

---

##  Visualization

- Random sample images from dataset with labels.
- Accuracy and loss trends (if training visualized over epochs).
- Optional: Confusion matrix plot using seaborn or matplotlib.

---

##  Notes

- Designed to run on Google Colab for ease of use.
- Ensure the dataset ZIP file follows the correct folder structure.
- Can be extended by applying other ensemble techniques (Random Forest, AdaBoost).

---

##  Author

- AMRUTHA A(20221ISE0045)
  ABINAYA K(20221ISE0007)
- Institution Name-PRESIDENCY UNIVERSITY


---

##  Future Enhancements

- Deploy as a web or mobile app using Flask/Streamlit or React Native.
- Extend to multi-crop disease classification using larger datasets.
- Integrate with IoT devices for real-time field monitoring.

---
