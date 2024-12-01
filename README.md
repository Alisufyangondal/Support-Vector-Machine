# Support-Vector-Machine
A comprehensive guide to machine learning model SVM
Tutorial: Implementing Support Vector Machines (SVM) for Breast Cancer Classification
Table of Contents
Tutorial: Implementing Support Vector Machines (SVM) for Breast Cancer Classification	1
Introduction	2
Dataset Description	2
Understanding Support Vector Machines (SVM)	3
Resilience and Generalization	4
Regularization and Soft Margin	4
Step-by-Step Implementation	5
Results	7
Performance Indicators	7
Major findings:	8
Conclusion	8
References:	8













Introduction
This tutorial explains how Support Vector Machines are used in classifying cases of breast cancer based on diagnostic attributes (Liu, 2018). This tutorial is best suited for beginners and intermediate learners in the domain of machine learning. We'll talk about the basic concepts related to SVM, shed light on the importance of the dataset, and step-by-step implementation using Python. In this tutorial, you will end up learning how to preprocess your data, train an SVM model, and evaluate its performance using visualizations and metrics. Proper classification of breast cancer as either benign or malignant in the right and timely manner requires classification (Karatzoglou et al., 2004). In the recent past, machine learning algorithms have been major assets to medical diagnostics; SVM algorithms are one such example. With the assistance of such algorithms, health care givers make swift and effective decisions that determine the patient's outcome. This tutorial uses the publicly available dataset Breast Cancer Wisconsin (Diagnostic), widely used within the machine learning community for classification in the context of cancer cases (Wang, 2020).
Dataset Description
The dataset used in this assignment is the Breast Cancer Wisconsin (Diagnostic) dataset, and it is publically available for download at Kaggle. GLOBALLY, breast cancer accounts for a very large percentage of cancer incidences, and early detection is known to greatly help in improving patients' outcome (Saha et al., 2014). This data enables machine learning practitioners to develop models that can support doctors identify tumors early so as many lives can be saved. It forms images from fine needle aspirates of the breast mass (Guyon et al., 2002). For each patient it, comprises of several numerical features that describe the characteristic of the cell nuclei.Here, some characteristics describing this dataset:

Total number of cases: 569 patients
There are 30 quantitative attributes.
 These are radius, texture, perimeter, area, etc.
 The features come from the measurement of cell nuclei in breast mass biopsies. 
 

Target variable:
•	M (Malignant): Cancerous tumour.
•	B. Benign. Non-malignant tumour.
Understanding Support Vector Machines (SVM)
One of the most commonly used robust algorithms in supervised learning is Support Vector Machines (SVM). SVM is mainly employed in classification (Wang, 2020). Even though it can be applied to regression analysis, SVM is primarily employed in classification where it is performing out of the world in finding separations between categories in complex high-dimensional spaces.

Central Idea: Hyperplanes and Margins In SVM, a hyperplane is the decision boundary dividing the feature space into different classes. In two-dimensional space, this hyperplane can also be termed as a line, and in higher dimensions as a hyperplane (Karatzoglou et al., 2004). SVM seeks a hyperplane that does not just distinguish the data points from one class into the other but also maximizes the margin between the closest points to each class. Such closest points are known as support vectors. The margin is defined as the distance between the hyperplane and the nearest data points of either class •  (Saha et al., 2014)
•  . 

Key Features of SVM
There are many key features and variants of SVM that make it a powerful tool for classification tasks:
•	Linear SVM
Linear SVMs are used when data is linearly separable. That means a two-dimensional space has a straight line, or a more than two-dimensional space has a hyperplane, that completely separates between the classes. The algorithm works by finding the best-fit hyperplane that maximizes the margin between the respective classes (Wang, 2020). If the data is separable with a linear boundary, then it's a very simple yet powerful technique: computationally very inexpensive.
•	Kernel SVM
Linear SVMs do not perform well when the data is not linearly separable. However, SVMs can still be used by applying kernel functions. A kernel function is a mathematical function that maps the input data into a higher-dimensional space where it becomes easier to separate classes with a hyperplane   (Saha et al., 2014). 

Linear Kernel: This is essentially equivalent to the standard linear SVM, with no transformation applied; hence it classifies the features with a line or hyperplane.

Polynomial Kernel: This kernel allows a polynomial function of the features to be the decision boundary that actually helps classify data separable by a polynomial surface.

Radial Basis Function Kernel: Actually, RBF is one of the most common kernels used in practice. It maps data into infinite dimension.



Resilience and Generalization
SVMs are very robust, especially in high-dimensional spaces. Unlike most of the other machine learning algorithms that suffer from the curse of dimensionality, SVMs work well with high-dimensional data. The reason is that SVM functions on a small number of support vectors rather than the entire dataset and hence the algorithm is scalable and less prone to overfitting (Karatzoglou et al., 2004). The SVM algorithm is also very efficient when the number of features is much more than that of the number of data points. This happens very frequently in many of the modern problems of machine learning, specially in bioinformatics, image processing, and natural language processing (Chang & Lin, 2011). For instance, the Breast Cancer Wisconsin dataset has 30 features; it is relatively high if compared with the number of samples (569), hence it becomes a perfect example for SVM. Moreover, since the SVM finds the maximum-margin model, there is a high guarantee that the model will generalize well to unseen data (Karatzoglou et al., 2004). 
Regularization and Soft Margin
Noisy or misclassified data points are some of the difficulties associated with the classification problem. Given this, SVM defines the soft-margin. Unlike hard-margin SVM that pushes all the wrong-class data points to at least one side of its margin, soft-margin SVM allows some of it to lie on the inappropriate side of the margin of the classifier. This is controlled by a single regularization parameter  (Liu, 2018). The chance to change the measurements of the sides by C The C-parameter makes SVM robust and able to manage noisy data. The following heat map shows the feature correlations of the data set entities (Bishop, 2006).
 

Why Use SVM? 
There are many reasons why SVM is the algorithm of choice for classification tasks: 
SVM has proved very effective in domains where the dimensions are too huge; it specially performs very well when the dimensions are much greater than data points, that happens frequently in many challenges faced by mankind (Karatzoglou et al., 2004). 

•	Memory efficiency: SVM uses only support vectors to decide about the decision boundary. SVM hence is computationally inexpensive not requiring the entire database for storage in memory (Cortes & Vapnik, 1995). 
•	Flexibility: SVMs can easily adjust to different types of data by choosing the appropriate kernel function. They can thereby adapt to handle both linear and nonlinear classification problems quite effectively. 
•	Good generalization ability: Since the SVM uses the margin maximization approach for all classes, its optimization method allows it to generalize well without risking overfitting.
 Step-by-Step Implementation

Step 1: Import Required Libraries
First step ias to import the important libraries for data processing, visualization and model building.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


Step 2: Data Loading:
Load the data for the dataset and print the  data information and some rows  from the dataset

# Load dataset from Google Drive
file_path = '/content/drive/MyDrive/Dataset/Breast Cancer Wisconsin (Diagnostic).csv'
data = pd.read_csv(file_path)

# Display basic information and the first few rows
print(data.info())


Step 3: Printing Data Head:
Write the following code to print the dataset head:
print(data.head())

Step 4: Data Cleaning and Preprocessing:

•	Remove columns like id and Unnamed: 32 that do not help in the classification process.
•	Encode Target Variable: Diagnosis column to numeric values: M -> 1, B -> 0.
# Drop unnecessary columns
data = data.drop(columns=['id', 'Unnamed: 32'])

# Encode the 'diagnosis' column (M -> 1, B -> 0)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])  # M = 1, B = 0


Step 5: Feature-Target Separation
Separate the dataset into features (X) and the target variable (y) dependent and independent features.
# Separate features and target variable
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']                # Target

Step 7: Feature Scaling

Standardizing the data is a very essential step for setting  SVM. It ensure that all the features are on the same scale and improving model performance.
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




Step 8: Train-Test Split
Now split the data into testing and training sets, divide it into 30% and 70 %.

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

Step 9: Train the SVM Model
 For the classification of the dataset we will use a linear kernel and a random state of 42.

# Train SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)


Step 10: Model Evaluation
Now, we will evaluate the model using such metrics as accuracy, confusion matrix, and classification report.
# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)


Step 11: Confusion Matrix Heatmap
. Confusion Matrix Heatmap
conf_matrix_df = pd.DataFrame(conf_matrix, index=["Benign (Actual)", "Malignant (Actual)"], 
                              columns=["Benign (Predicted)", "Malignant (Predicted)"])
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.show()


 


Results
Performance Indicators
Accuracy: 97.66%
Confusion Matrix:
True negatives: 106
False Positives: 2
False Negative: 2
True Positives: 61

Major findings:
•	The performance was excellent with minimal misclassification.
•	Most benign and malignant cases were correctly classified, which is a high reliability.



Conclusion

In this tutorial, we developed an SVM model that classifies breast cancer into benign or malignant classes. We addressed the whole workflow from data preprocessing, visualization, and evaluation. The excellent accuracy achieved demonstrates the feasibility of SVM in healthcare applications. They are one of the most adaptable classes of algorithms, in general, for classification purposes. It really did outstand in a highly-dimensional environment, which let very linear and very nonlinear sets be handled with their variety of kernel functions. SVMs are good generalizers and thus ideal for any medical diagnosis problem like breast cancer classification into malignant or benign, focusing on maximizing the margin between classes. If proper kernel selection and regularization are used, then it has been found that SVMs work very well for most datasets. After optimization, such models can be utilized in real-world applications for the necessary support in early diagnosis and detection of cancer.



References:
1.	Cortes, C. & Vapnik, V., 1995. Support-vector networks. Machine learning, 20(3), pp.273-297.
2.	Bishop, C.M., 2006. Pattern recognition and machine learning. Springer.
3.	Karatzoglou, A., Smola, A., Hornik, K. & Zeileis, A., 2004. kernlab - An S4 package for kernel methods in R. Journal of Statistical Software, 11(9), pp.1-20.
4.	Zhang, Z., 2016. Introduction to machine learning. Springer Science & Business Media.
5.	Hsu, C.-W., Chang, C.-C. & Lin, C.-J., 2010. A practical guide to support vector classification. Department of Computer Science, National Taiwan University.
6.	Wang, L., 2020. Breast cancer classification using support vector machine (SVM) and deep learning techniques. International Journal of Computer Applications, 975(5), pp.13-19.
7.	Liu, L., 2018. Cancer classification with support vector machine using feature selection and optimization techniques. Computers in Biology and Medicine, 104, pp. 63-72.
8.	Chang, C.-C. & Lin, C.-J., 2011. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2(3), pp.1-39.
9.	Saha, S., Biswas, S., & Das, S., 2014. Breast Cancer Detection Using Support Vector Machine. International Journal of Computer Science and Network Security, 14(9), pp.1-8.
10.	Guyon, I., Weston, J., Barnhill, S. & Vapnik, V., 2002. Gene selection for cancer classification using support vector machines. Machine Learning, 46(1-3), pp.389-422.

