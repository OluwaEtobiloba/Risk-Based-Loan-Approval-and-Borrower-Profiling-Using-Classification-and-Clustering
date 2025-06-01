# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:00:40 2025

@author: Oluwatobiloba Alao
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#Import dataset into the environment
d1 = pd.read_csv("C:/Users/user/Desktop/projects/Loan.csv")
d1.head()
d1.shape

#Let examine our target variable 
unique_vals = d1['LoanApproved'].unique()
print("Unique LoanApproved values:", unique_vals)

#Map the values of the target variable to the right binary since the data contains binary vaibale
d1_clean = d1[d1['LoanApproved'].notna()].copy()
mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}
d1_clean['target'] = d1_clean['LoanApproved'].map(mapping)
d1_clean = d1_clean[d1_clean['target'].notna()]

#create a matrrix for the features dropping the loanapproved, datatime and target variables.
#all other columns in the dataset used for the the training of the model
X = d1_clean.drop(['LoanApproved', 'ApplicationDate', 'target'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = d1_clean['target'].astype(int)

#train and test split the dataset and set the scale for the features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Classification Model using logistics regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

result = classification_report(y_test, y_pred, output_dict=True)
result_df = pd.DataFrame(result).transpose()
print(result_df)
print("ROC AUC:", roc_auc_score(y_test, y_prob))

#Build out the confusion matrix for the classification result 
matrix = confusion_matrix(y_test, y_pred)
matrix_df = pd.DataFrame(matrix, index=['Actual_N', 'Actual_Y'], columns=['Pred_N', 'Pred_Y'])
print(matrix_df)

plt.figure()
plt.imshow(matrix_df, interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.yticks([0, 1], ['Not Approved', 'Approved'])
thresh = matrix.max() / 2
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, str(matrix[i, j]),
                 ha='center',
                 color='white' if matrix[i, j] > thresh else 'black')
plt.tight_layout()
plt.show()

#Build an unsupervised learning model using K-means clustering 
X_full_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

sil_score = silhouette_score(X_pca, clusters)
print("Silhouette Score:", sil_score)

#Plot our clusters
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clusters on PCA-reduced Data')
plt.show()

#give the cluster a profile 
d1_cluster = d1_clean.copy()
d1_cluster['cluster'] = clusters
profile = d1_cluster.drop(['ApplicationDate', 'LoanApproved', 'target'], axis=1)
profile = pd.get_dummies(profile, drop_first=True)
profile = profile.groupby(d1_cluster['cluster']).mean()
print(profile)