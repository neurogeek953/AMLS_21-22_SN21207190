# AMLS_21-22_SN21207190: MRI-Tumor-Detection

1) Project Summary

This project consisted of two tasks:
- Task A: Identify if a tumor is in an MRI scan.
- Task B: Identify if there is a tumor in the MRI scan and if this tumor is a meningioma, glioma or pituitary tumor.
Task A was solved with linear classifiers and the deep learning models in the files:
-> classifier_1_OUT_XDROP.h5
-> classifier_1_OUT.h5
-> classifier_2_OUT_XDROP.h5
-> classifier_2_OUT.h5
-> classifier_2_OUT_100.h5
Task B was solved with the deep learning models in the files
-> classifier_4_OUT_XDROP.h5
-> classifier_4_OUT.h5

2) File organisation in Alphabetical Order 

- classifier_1_OUT.h5: CNN with 1 sigmoid output node trained with dropout for 50 Epochs
- classifier_1_OUT_XDROP.h5: CNN with 1 sigmoid output node trained without dropout for 50 Epochs
- classifier_2_OUT.h5: CNN with 2 SoftMax output nodes trained with dropout for 50 Epochs
- classifier_2_OUT_XDROP.h5: CNN with 2 SoftMax output nodes trained without dropout for 50 Epochs
- classifier_2_OUT_100.h5: CNN with 2 SoftMax output nodes trained with dropout for 100 Epochs
- classifier_4_OUT.h5: CNN with 4 SoftMax output nodes trained with dropout for 50 Epochs
- classifier_4_OUT_XDROP.h5: CNN with 4 SoftMax output nodes trained with dropout for 50 Epochs
- dataset.zip: The Training Set (Absent from GitHub repository due to upload limit) -> see NOTE 1 and NOTE 3
- CNN.py: The CNN_Classifier Class
- ExploratoryDataAnalysis.py: Execute Exploratory Data Analysis (EDA) on the Training and Test Sets | Imports the preprocess_data function from TrainCNN.py
- LinearClassifiers.py: The LinearClassifiers Class
- LogisticRegression.py: The LogisticRegressor Class
- PrincipleComponentAnalysis.py: The PrincipleComponents Class
- SVMClassifiers.py: The SupportVectorClassifier Class
- test.zip: The Test Set
- TestCNN.py: Test  all 7 CNNs
- TrainCNN.py: Train all 7 CNNs
- TrainLinearClassifiers.py: Train and Test the 6 Linear Classifiers

NOTE 1 -> The dataset folder is unavailable as Github will not allow to upload a file of this size but it can be downloaded from the following kaggle page: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri?select=Training

NOTE 2 -> The Diagram of the File Organisation is on "Figure 21. Implementation's File Organisation" of the report.

NOTE 3 -> All the files are available at the following Google Drive link: https://drive.google.com/file/d/1NikSAw3ReqtoNsZ_tsra_sYxkXrg6GsZ/view?usp=sharing

3) Libraries, Packages and their versions

- NumPy == 1.18.5
- scikit-learn == 0.24.2
- imbalanced-learn == 0.8.1
- tensorflow == 2.3.0
- OpenCV —> opencv-python == 4.5.3.56 

4) Instruction to Run the Files
- Download the dataset from Kaggle before training the models and unzip the all the files
- Observe the Distribution of the Binary and Original Labels by running the file xploratoryDataAnalysis.py 
- Load and Test the 6 CNNs presented in the report by running TestCNN.py
- Train and Test the 6 CNNs by running the file TestCNN.py
- Train and Test the 6 Linear Classifiers by running the file TrainLinearClassifiers.py

