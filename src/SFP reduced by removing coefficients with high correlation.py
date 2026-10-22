#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
from nltk import DecisionTreeClassifier
from nltk.classify.svm import SvmClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load the four datasets
df1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-KC1(updated).csv")
df2 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MW1(updated).csv")
df3 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-PC5(updated).csv")
df4 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MC1(updated).csv")


# In[3]:


# Displaying first few rows of each dataset
print("Dataset 1:")
print(df1.head())

print("\nDataset 2:")
print(df2.head())

print("\nDataset 3:")
print(df3.head())

print("\nDataset 4:")
print(df4.head())


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix heatmap for each dataset
for i, df in enumerate([df1, df2, df3, df4], start=1):
    plt.figure(figsize=(20, 16))  # Significantly larger figure

    # Generate a correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap with even smaller annotation font
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        annot_kws={"size": 13}  # Smaller text for tight spaces
    )
    #plt.title(f'Dataset {i} Correlation Matrix after removing highly correlated attributes', fontsize=16)
    plt.title(f'Dataset {i} Correlation Matrix after removing highly correlated attributes', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


# In[5]:


# List of datasets and their names
datasets = [(df1, "KC1"), (df2, "MW1"), (df3, "PC5"), (df4, "MC1")]
threshold = 0.9  # Positive correlation threshold

for df, name in datasets:
    print(f"\n🔍 Highly positively correlated features in {name} (correlation ≥ {threshold}):")

    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()  # <-- no .abs()

    # Exclude self-correlations using upper triangle only
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs with correlation ≥ threshold
    high_corr = [
        (col1, col2, upper.loc[col1, col2])
        for col1 in upper.columns
        for col2 in upper.columns
        if upper.loc[col1, col2] >= threshold
    ]

    if high_corr:
        for col1, col2, value in sorted(high_corr, key=lambda x: -x[2]):
            print(f"  {col1} ↔ {col2} : {value:.2f}")
    else:
        print("  No highly positively correlated pairs found.")


# In[6]:


from sklearn.impute import SimpleImputer


def preprocess_data(dataset, target_column):
    # Drop rows with missing target values (if any)
    dataset = dataset.dropna(subset=[target_column])

    # Separating the features (X) and target (y)
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (scale the data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred.round(), average='weighted', zero_division=1)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return precision, recall, f1, mse, r2

dataset_kc1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-KC1(updated).csv")
dataset_mw1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MW1(updated).csv")
dataset_pc5 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-PC5(updated).csv")
dataset_mc1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MC1(updated).csv")

X_train_kc1, X_test_kc1, y_train_kc1, y_test_kc1 = preprocess_data(dataset_kc1, 'Defective')
X_train_mw1, X_test_mw1, y_train_mw1, y_test_mw1 = preprocess_data(dataset_mw1, 'Defective')
X_train_pc5, X_test_pc5, y_train_pc5, y_test_pc5 = preprocess_data(dataset_pc5, 'Defective')
X_train_mc1, X_test_mc1, y_train_mc1, y_test_mc1 = preprocess_data(dataset_mc1, 'Defective')

results_kc1 = {}
results_mw1 = {}
results_pc5 = {}
results_mc1 = {}

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM Regression': SVC(kernel='linear'),
    'Decision Tree Classification': DecisionTreeClassifier(criterion='entropy'),
    'Random Forest Classification': RandomForestClassifier(random_state=42, n_estimators=100)
}

for model_name, model in models.items():
    model.fit(X_train_kc1, y_train_kc1)
    y_pred = model.predict(X_test_kc1)
    results_kc1[model_name] = evaluate_model(y_test_kc1, y_pred)
    
for model_name, model in models.items():
    model.fit(X_train_mw1, y_train_mw1)
    y_pred = model.predict(X_test_mw1)
    results_mw1[model_name] = evaluate_model(y_test_mw1, y_pred)
    
for model_name, model in models.items():
    model.fit(X_train_pc5, y_train_pc5)
    y_pred = model.predict(X_test_pc5)
    results_pc5[model_name] = evaluate_model(y_test_pc5, y_pred)
    
for model_name, model in models.items():
    model.fit(X_train_mc1, y_train_mc1)
    y_pred = model.predict(X_test_mc1)
    results_mc1[model_name] = evaluate_model(y_test_mc1, y_pred)
    
results_kc1_df = pd.DataFrame(results_kc1, index=['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score'])
results_mw1_df = pd.DataFrame(results_mw1, index=['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score'])
results_pc5_df = pd.DataFrame(results_pc5, index=['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score'])
results_mc1_df = pd.DataFrame(results_mc1, index=['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score'])

# Print results in tabular form
print("KC1 Dataset Results:")
print(results_kc1_df.to_markdown())

print("\nMW1 Dataset Results:")
print(results_mw1_df.to_markdown())

print("\nPC5 Dataset Results:")
print(results_pc5_df.to_markdown())

print("\nMC1 Dataset Results:")
print(results_mc1_df.to_markdown())


# In[ ]:





# In[7]:


import numpy as np

# Define the metrics and datasets
metrics = ['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score']
datasets = {
    'KC1': results_kc1_df,
    'MW1': results_mw1_df,
    'PC5': results_pc5_df,
    'MC1': results_mc1_df
}

# Define colors for each dataset
colors = {
    'KC1': 'b',
    'MW1': 'r',
    'PC5': 'g',
    'MC1': 'y'
}

# Plot each metric
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = results_kc1_df.columns.tolist()  # Assumes same columns for all datasets
    x = np.arange(len(model_names))  # the label locations
    bar_width = 0.2

    for i, (dataset_name, results_df) in enumerate(datasets.items()):
        values = results_df.loc[metric]
        offset = (i - 1.5) * bar_width  # Position bars side by side
        ax.bar(x + offset, values, width=bar_width, color=colors[dataset_name], alpha=0.7, label=dataset_name)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison of {metric} for Different Models")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# In[8]:


def plot_roc_curves(dataframes, target_column='Defective'):
    plt.figure(figsize=(16, 12))

    for i, (name, df) in enumerate(dataframes.items(), 1):
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Impute & scale
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Random Forest classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # ROC values
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # 🔵 Print AUC value
        print(f"AUC for {name}: {roc_auc:.4f}")

        # Plot
        plt.subplot(2, 2, i)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'b--', lw=1)  # Blue dotted diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

dataframes = {
    "KC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-KC1(updated).csv"),
    "MW1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MW1(updated).csv"),
    "PC5": pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-PC5(updated).csv"),
    "MC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data_New\csv_result-MC1(updated).csv")
}

plot_roc_curves(dataframes)


# In[ ]:




