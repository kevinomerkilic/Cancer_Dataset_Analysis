Breast Cancer Classification Using Machine Learning: A Diagnostic Approach
Author: Omer Kilic
Project Title: Breast Cancer Prediction with Classical ML Algorithms
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (569 samples, 30 features, 2 classes)
Tool: Python with Scikit-Learn, Google Colab

1. Abstract
Breast cancer is among the most life-threatening conditions affecting women globally. Early and
accurate diagnosis is critical for successful treatment and survival. In this project, we implement
and compare multiple machine learning models—Logistic Regression, Support Vector Machines
(SVM), K-Nearest Neighbors (KNN), and Random Forests—to classify breast tumors as
malignant (cancerous) or benign (non-cancerous) based on cellular characteristics extracted
from digitized images of breast tissue.

3. Introduction
Traditional diagnostic methods like mammography and biopsy, while effective, are time-
consuming and prone to human error. With the rise of computational diagnostics, machine
learning models now offer scalable, accurate, and reproducible alternatives that can significantly
reduce the diagnostic workload on medical professionals.
This study leverages the Wisconsin Breast Cancer Diagnostic Dataset, a benchmark dataset
provided by the UCI repository, which includes 30 real-valued features calculated from digitized
images of fine needle aspirates (FNAs).

5. Data Preprocessing and Exploration
• Dataset: 569 entries with 30 features + target (0 = Malignant, 1 = Benign)
• No missing values were found
• Features scaled using StandardScaler for optimal model convergence
• Benign (1) cases are more frequent than malignant (0), creating a slight class imbalance
A correlation heatmap revealed high collinearity among radius, perimeter, and area-related
features, justifying potential for dimensionality reduction or feature selection in further studies.

6. Machine Learning Models and Performance
Four models were trained on the dataset (80% training, 20% testing):
Model Accuracy
(%) Comments
SVM 98.25% Best overall performance
Logistic
Regression 97.36% Very high precision/
recall
Random Forest 95.61% Robust to overfitting
KNN 94.74% Slightly lower accuracy
Confusion Matrix (Logistic Regression):
[[41 2]
[ 1 70]]
• True Positives: 70
• True Negatives: 41
• Precision: 0.97+
• Recall: 0.99+
• F1 Score: 0.97+
ROC AUC:
• Logistic Regression AUC: 1.00
• Random Forest AUC: 0.99+
These results show an exceptional capacity to correctly classify malignant vs benign tumors
with minimal false positives or false negatives.

7. Impact and Real-World Relevance
The model you’ve developed has real-world significance in the field of medical diagnostics
and public health:
• Clinical Decision Support: Your model can assist pathologists and radiologists in
interpreting complex biopsy data, reducing diagnostic errors.
• Faster Diagnosis: Automated classification enables rapid screening, critical in resource-
constrained healthcare systems.
• Cost Savings: It reduces the need for unnecessary biopsies and follow-up tests.
• Global Health Equity: Once deployed, lightweight ML systems like this can provide
diagnostic support in underserved regions where medical specialists are limited.
With its high accuracy and interpretability, especially with models like Logistic Regression and
Random Forest, this system can be integrated into healthcare pipelines for scalable cancer
screening.

8. Conclusion
This project demonstrates the effectiveness of classical machine learning techniques in solving
real-world biomedical problems. SVM achieved the best performance, but all models displayed
excellent diagnostic capability.
Moving forward, additional improvements could include:
• Hyperparameter tuning (e.g., GridSearchCV)
• Feature selection (e.g., PCA or mutual information)
• Model interpretability tools (e.g., SHAP, LIME)
• Integration into cloud-based diagnostic platforms (e.g., AWS HealthLake)
Appendix / References
• UCI Machine Learning Repository – Breast Cancer Wisconsin Dataset
• Scikit-Learn Documentation
• “Machine Learning in Medical Diagnosis” – Nature Biomedical Engineering

******************************************************************************

Additional Information and Further Development
The Breast Cancer Wisconsin Diagnostic Dataset provided by scikit-learn is
preprocessed and doesn't include original images or visual annotations. It only gives you the
numeric features extracted from images via a digitized process (like radius, texture, area, etc.).
But you can access the raw version of this dataset (with more context, labels, and sometimes
images) in other sources like UCI Machine Learning Repository or Kaggle.

1. UCI Machine Learning Repository (Official Source)
• URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
• Description: Includes original data, label description, and feature extraction process
• Files:
◦ wdbc.data → raw CSV file
◦ wdbc.names → explains how each column was derived

2. Kaggle Datasets with Images
Kaggle has several versions of breast cancer datasets, including:
IDC Breast Cancer Histology Dataset (with images!)
• https://www.kaggle.com/paultimothymooney/breast-histopathology-images
• Contains 277,524 microscopic images
• Labels: IDC Positive (Invasive Ductal Carcinoma) vs Negative
BreakHis Dataset (8,000+ Images)
• https://www.kaggle.com/ambarish/breakhis
• Different magnification levels (40x, 100x, 200x, 400x)
• Malignant/Benign labels

Labeling Info (for Scikit-Learn Dataset)
From load_breast_cancer() in Scikit-Learn:
• target_names: ['malignant', 'benign'] → i.e., 0 = malignant, 1 =benign
• feature_names: 30 numerical features like:
◦ mean radius, mean texture, worst concave points, etc.
◦ Derived from digitized image contours
We can explore this in code:
python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data.feature_names) # Feature list
print(data.target_names) # ['malignant', 'benign']

Actual Images?
We can build a Convolutional Neural Network (CNN) using histology images from Kaggle.
We can:
• Download and load the image dataset
• Preprocess images (resize, normalize)
• Build a CNN with TensorFlow/Keras
• Train a model to classify tumors visually

LESS-TECHNICAL DESCRIPTION
🎭 The Story of the Cancer Dataset: A Medical Detective Mystery
Imagine we're Dr. Data, a data scientist in a futuristic hospital. Our mission? To help doctors
detect whether a tumor is benign (harmless) or malignant (dangerous) using a powerful
microscope that captures 30 different features of each tumor cell.
These features are things like:
• Size of the nucleus (like how big the cell’s core is),
• Smoothness of the cell (is it rough or silky?),
• Compactness, texture, symmetry… etc.

🧪 The Dataset = Your Lab Report
Each patient has a report with these features. We get hundreds of them. But we don’t know what
these numbers mean just by looking — so our job is to train a smart assistant (a machine
learning model) to find patterns between the features and the diagnosis (B for benign, M for
malignant).

🔍 Step 1: Exploring the Dataset (Like Sherlock Holmes)
We open the dataset and start asking questions like:
• Which features are more common in malignant tumors?
• Are there outliers or weird values?
• Do some features always go together (correlated)? This is like Sherlock checking for
footprints, fingerprints, and motive.

📊 Step 2: Visualizing Data (Creating the Crime Scene Map)
We create plots, graphs, and heatmaps to help visualize relationships. For example:
• A boxplot might show that malignant tumors have much higher radius values.
• A heatmap helps you see which features move together (like “texture” and
“smoothness”).
It’s like drawing a map of the crime scene with red pins showing where clues lie.

🧠 Step 3: Training a Model (Hiring a Smart Assistant)
We pick a machine learning algorithm like Logistic Regression, Decision Tree, or Random
Forest. This assistant learns from past patient cases — what symptoms led to cancer — and then
predicts if a new tumor is benign or malignant.
Think of this like training a medical intern to learn from your files and make predictions. You
test them, tweak their learning (hyperparameters), and measure how well they do using
accuracy, precision, recall, F1 score.

🎯 Final Step: Making Predictions (Saving Lives!)
Once our model is accurate, it becomes like a trusted diagnostic assistant. When a new patient
comes in, you feed in the 30 features and ask:
"Based on your training, do you think this tumor is benign or malignant?"
If it says malignant, the doctors take fast action. If benign, less worry. Our model becomes a
tool that helps save lives.

🎬 Analogy Wrap-up:
• 🕵 We = Detective Doctor (data scientist)
• 📁 Dataset = Medical files with 30 symptoms (features)
• 🔍 EDA = Looking for patterns & red flags
• 📊 Visuals = Drawing diagrams and charts
• 🤖 ML model = Your trained assistant intern
• 🧪 Testing = Quizzing the intern on old cases
• 🏥 Prediction = Using the intern to detect future cases
