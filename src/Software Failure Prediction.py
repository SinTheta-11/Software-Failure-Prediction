#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

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


# # Data analysis part

# Purpose: In this step, we load the four datasets into separate Pandas DataFrames. This allows us to independently analyze and manipulate each dataset. By displaying the first few rows, we verify that the data has been loaded correctly and that we can view the structure of each dataset.

# In[2]:


# Load the four datasets
df1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-KC1.csv")
df2 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MW1.csv")
df3 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-PC5.csv")
df4 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MC1.csv")


# Purpose : We check the structure of each dataset using info(). This step gives us an understanding of the number of columns, data types, and any missing values in each dataset. It's essential to know what data we're working with and whether any columns need special attention (e.g., missing or improperly formatted values).

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


# In[4]:


# Check dataset information for each dataset
print("Dataset 1 Information:")
df1.info()

print("\nDataset 2 Information:")
df2.info()

print("\nDataset 3 Information:")
df3.info()

print("\nDataset 4 Information:")
df4.info()


# Purpose : This step identifies any missing values in the datasets. Missing values can lead to erroneous results, so it's important to determine how many missing entries are present in each column of each dataset before proceeding with further analysis.

# In[5]:


# Check for missing values in each dataset
print("Missing values in Dataset 1:")
print(df1.isnull().sum())

print("\nMissing values in Dataset 2:")
print(df2.isnull().sum())

print("\nMissing values in Dataset 3:")
print(df3.isnull().sum())

print("\nMissing values in Dataset 4:")
print(df4.isnull().sum())


# Purpose: After identifying missing values, we handle them by filling in the missing values with the mean of the respective column. This is a standard approach to dealing with missing data, ensuring that we don't lose valuable data rows while avoiding introducing bias.

# In[6]:


import re
# Function to clean non-numeric characters from a string
def clean_non_numeric(data):
    if isinstance(data, str):
        return pd.to_numeric(re.sub(r'[^0-9.]', '', data), errors='coerce')
    return pd.to_numeric(data, errors='coerce')


# In[7]:


# Fill missing values with the mean for each dataset
df1.fillna(df1.mean(), inplace=True)
df2.fillna(df2.mean(), inplace=True)
df3.fillna(df3.mean(), inplace=True)
df4.fillna(df4.mean(), inplace=True)


# In[8]:


# Verify missing values have been filled
print("Missing values after handling in Dataset 1:")
print(df1.isnull().sum())
print("Missing values after handling in Dataset 2:")
print(df2.isnull().sum())
print("Missing values after handling in Dataset 3:")
print(df3.isnull().sum())
print("Missing values after handling in Dataset 4:")
print(df4.isnull().sum())


# Purpose: Generating descriptive statistics helps us understand the basic statistical properties of the dataset, such as the mean, median, standard deviation, and percentiles. This gives us insights into the central tendencies and the spread of the data.

# In[9]:


# Generate descriptive statistics for each dataset
print("\nDescriptive Statistics for Dataset 1:")
print(df1.describe())

print("\nDescriptive Statistics for Dataset 2:")
print(df2.describe())

print("\nDescriptive Statistics for Dataset 3:")
print(df3.describe())

print("\nDescriptive Statistics for Dataset 4:")
print(df4.describe())


# Purpose: Boxplots are useful for visually identifying outliers. Outliers are extreme values that can distort statistical analyses, so it’s crucial to detect and manage them. This step creates boxplots for each dataset, allowing us to see any values that are significantly higher or lower than the rest of the data.

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create vertically aligned subplots: 4 rows, 1 column
fig, axs = plt.subplots(4, 1, figsize=(20, 60))  # Larger and tall layout
fig.suptitle('Boxplots of Datasets', fontsize=22)

# Plot boxplot for Dataset 1
sns.boxplot(data=df1, ax=axs[0])
axs[0].set_title('Dataset 1 Boxplot', fontsize=16)
axs[0].tick_params(axis='x', rotation=90)

# Plot boxplot for Dataset 2
sns.boxplot(data=df2, ax=axs[1])
axs[1].set_title('Dataset 2 Boxplot', fontsize=16)
axs[1].tick_params(axis='x', rotation=90)

# Plot boxplot for Dataset 3
sns.boxplot(data=df3, ax=axs[2])
axs[2].set_title('Dataset 3 Boxplot', fontsize=16)
axs[2].tick_params(axis='x', rotation=90)

# Plot boxplot for Dataset 4
sns.boxplot(data=df4, ax=axs[3])
axs[3].set_title('Dataset 4 Boxplot', fontsize=16)
axs[3].tick_params(axis='x', rotation=90)

# Adjust spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave space for suptitle
plt.show()


# Purpose: Outliers are removed using the Interquartile Range (IQR) method. This method identifies and excludes values that lie outside the range of typical data points (beyond 1.5 times the IQR). Removing outliers can improve the accuracy of our models and prevent skewed analyses.

# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

# 1) Make the figure tall enough that each subplot is roughly square:
#    With 4 rows × 2 columns, each subplot is ~figsize[0]/2 wide and figsize[1]/4 tall.
#    To get width≈height, set figsize[1] ≈ figsize[0]*(4/2) = figsize[0]*2
fig, axs = plt.subplots(4, 2, figsize=(24, 48))  # width=24, height=48

fig.suptitle('Outlier Removal Comparison (Boxplots)', fontsize=24)

datasets = [df1, df2, df3, df4]
dataset_names = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']

for row, (df, name) in enumerate(zip(datasets, dataset_names)):
    # Before removing outliers
    ax_before = axs[row, 0]
    sns.boxplot(data=df, ax=ax_before)
    ax_before.set_title(f'{name} – Before', fontsize=16)
    ax_before.tick_params(axis='x', rotation=90, labelsize=10)
    
    # 2) Force square axes:
    ax_before.set_box_aspect(1)

    # Remove outliers via IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # After removing outliers
    ax_after = axs[row, 1]
    sns.boxplot(data=df_cleaned, ax=ax_after)
    ax_after.set_title(f'{name} – After', fontsize=16)
    ax_after.tick_params(axis='x', rotation=90, labelsize=10)
    
    # Force square axes here as well
    ax_after.set_box_aspect(1)

# Increase space between subplots
plt.subplots_adjust(hspace=0.6, wspace=0.4)

# Tight layout but leave room for suptitle
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Purpose: In this step, we calculate the mean, median, and percentiles (Q1 and Q3) for each dataset. This helps us understand the distribution pattern of the data, such as whether it's normally distributed, skewed, or has a high level of variance. Understanding the data distribution is crucial for model selection and performance.

# In[12]:


# Iterate over datasets to calculate and plot mean, median, and percentiles
for i, df in enumerate([df1, df2, df3, df4], start=1):
    mean = df.mean()
    median = df.median()
    percentile_25 = df.quantile(0.25)
    percentile_75 = df.quantile(0.75)

    # Create a DataFrame for the statistics to plot
    stats_df = pd.DataFrame({
        'Mean': mean,
        'Median': median,
        '25th Percentile (Q1)': percentile_25,
        '75th Percentile (Q3)': percentile_75
    })

    # Plot the statistics for the current dataset
    plt.figure(figsize=(10, 6))
    stats_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Dataset {i} - Mean, Median, and Percentiles')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Purpose: The trimmed mean is calculated by excluding a certain percentage of extreme values from both ends of the data. This is useful when we want a measure of central tendency that is not affected by outliers. The trimmed mean provides a more robust estimate of the true center of the data.

# In[13]:


from scipy.stats import trim_mean

# Trimming fraction (10%)
trim_fraction = 0.1

# Iterate over datasets to calculate trimmed mean and other statistics, and plot them
for i, df in enumerate([df1, df2, df3, df4], start=1):
    # Calculate statistics
    mean = df.mean()
    median = df.median()
    percentile_25 = df.quantile(0.25)
    percentile_75 = df.quantile(0.75)

    # Calculate trimmed mean
    trimmed_mean = trim_mean(df, proportiontocut=trim_fraction)

    # Create a DataFrame for the statistics to plot
    stats_df = pd.DataFrame({
        'Mean': mean,
        'Median': median,
        'Trimmed Mean': trimmed_mean,
        '25th Percentile (Q1)': percentile_25,
        '75th Percentile (Q3)': percentile_75
    })

    # Plot the statistics for the current dataset
    plt.figure(figsize=(10, 6))
    stats_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Dataset {i} - Mean, Trimmed Mean, Median, and Percentiles')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Purpose: In this step, we perform correlation analysis to identify relationships between variables. A correlation matrix is created to show how strongly different variables are related to each other. This helps in understanding which features are most important and which ones are redundant.

# In[44]:


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
        annot_kws={"size": 9}  # Smaller text for tight spaces
    )
    plt.title(f'Dataset {i} Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()



# # Finding Highly Positively Correlated Features 

# In[15]:


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


# # Function to preprocess the dataset (handle missing values, scaling, train/test split)
# 

# In[16]:


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


# # Function to evaluate models using precision, recall, f1 score, and other metrics
# 

# In[17]:


def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred.round(), average='weighted', zero_division=1)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return precision, recall, f1, mse, r2


# # Load the datasets
# 

# In[18]:


dataset_kc1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-KC1.csv")
dataset_mw1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MW1.csv")
dataset_pc5 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-PC5.csv")
dataset_mc1 = pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MC1.csv")


# # Preprocess the KC1, JM1, PC5, MC1 datasets

# In[19]:


X_train_kc1, X_test_kc1, y_train_kc1, y_test_kc1 = preprocess_data(dataset_kc1, 'Defective')
X_train_mw1, X_test_mw1, y_train_mw1, y_test_mw1 = preprocess_data(dataset_mw1, 'Defective')
X_train_pc5, X_test_pc5, y_train_pc5, y_test_pc5 = preprocess_data(dataset_pc5, 'Defective')
X_train_mc1, X_test_mc1, y_train_mc1, y_test_mc1 = preprocess_data(dataset_mc1, 'Defective')


# # Dictionary to store results for both datasets

# In[20]:


results_kc1 = {}
results_mw1 = {}
results_pc5 = {}
results_mc1 = {}


# # Models to be applied
# 

# In[21]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM Regression': SVC(kernel='linear'),
    'Decision Tree Classification': DecisionTreeClassifier(criterion='entropy'),
    'Random Forest Classification': RandomForestClassifier(random_state=42, n_estimators=100)
}


# # Train and evaluate each model for the KC1 dataset
# 

# In[22]:


for model_name, model in models.items():
    model.fit(X_train_kc1, y_train_kc1)
    y_pred = model.predict(X_test_kc1)
    results_kc1[model_name] = evaluate_model(y_test_kc1, y_pred)


# # Train and evaluate each model for the MW1 dataset

# In[23]:


for model_name, model in models.items():
    model.fit(X_train_mw1, y_train_mw1)
    y_pred = model.predict(X_test_mw1)
    results_mw1[model_name] = evaluate_model(y_test_mw1, y_pred)


# # Train and evaluate each model for the PC5 dataset

# In[24]:


for model_name, model in models.items():
    model.fit(X_train_pc5, y_train_pc5)
    y_pred = model.predict(X_test_pc5)
    results_pc5[model_name] = evaluate_model(y_test_pc5, y_pred)


# # Train and evaluate each model for the MC1 dataset

# In[25]:


for model_name, model in models.items():
    model.fit(X_train_mc1, y_train_mc1)
    y_pred = model.predict(X_test_mc1)
    results_mc1[model_name] = evaluate_model(y_test_mc1, y_pred)


# # Convert results to DataFrame for easy comparison

# In[26]:


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


# # Plot the comparison of F1 Score for all 4 datasets
# 

# In[41]:


import numpy as np
import matplotlib.pyplot as plt

# List of models
models = results_kc1_df.columns.tolist()

# Bar width
bar_width = 0.1
x = np.arange(len(models))  # X-axis locations for the groups

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each dataset with offset for grouped bars
ax.bar(x - 1.5 * bar_width, results_kc1_df.loc['F1 Score'], width=bar_width, label='KC1', color='b', alpha=0.7)
ax.bar(x - 0.5 * bar_width, results_mw1_df.loc['F1 Score'], width=bar_width, label='MW1', color='r', alpha=0.7)
ax.bar(x + 0.5 * bar_width, results_pc5_df.loc['F1 Score'], width=bar_width, label='PC5', color='g', alpha=0.7)
ax.bar(x + 1.5 * bar_width, results_mc1_df.loc['F1 Score'], width=bar_width, label='MC1', color='y', alpha=0.7)

# X-axis setup
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.set_ylabel('F1 Score')
ax.set_title('Comparison of F1 Scores for Different Models')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()


# # Plot the comparison of Precision, Recall, F1 Score, MSE, and R2 Score for all 4 datasets
# 

# In[46]:


import numpy as np

# Define the metrics and datasets
metrics = ['Precision', 'Recall', 'F1 Score', 'MSE', 'R2 Score']
datasets = {
    'Dataset 1': results_kc1_df,
    'Dataset 2': results_mw1_df,
    'Dataset 3': results_pc5_df,
    'Dataset 4': results_mc1_df
}

# Define colors for each dataset
colors = {
    'Dataset 1': 'b',
    'Dataset 2': 'r',
    'Dataset 3': 'g',
    'Dataset 4': 'y'
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
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison of {metric} for Different Models")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# In[29]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define metric data for each dataset
metrics = ['Precision', 'Recall', 'F1 Score']
model_names = ['Logistic Regression', 'SVM Regression', 'Decision Tree Classification', 'Random Forest Classification']

# Create DataFrames for each dataset
kc1_data = {
    'Logistic Regression': [0.80115, 0.838863, 0.801022],
    'SVM Regression': [0.769925, 0.831754, 0.775415],
    'Decision Tree Classification': [0.817476, 0.831754, 0.823245],
    'Random Forest Classification': [0.831596, 0.853081, 0.834089]
}
mw1_data = {
    'Logistic Regression': [0.854701, 0.888889, 0.945534],
    'SVM Regression': [0.879825, 0.888889, 0.884206],
    'Decision Tree Classification': [0.901235, 0.901235, 0.901235],
    'Random Forest Classification': [0.854701, 0.888889, 0.945534]
}
pc5_data = {
    'Logistic Regression': [0.965785, 0.970913, 0.967578],
    'SVM Regression': [0.964001, 0.971495, 0.962171],
    'Decision Tree Classification': [0.976724, 0.975567, 0.976104],
    'Random Forest Classification': [0.973848, 0.97644, 0.974749]
}
mc1_data = {
    'Logistic Regression': [0.994748, 0.99472, 0.992842],
    'SVM Regression': [0.993704, 0.993664, 0.990506],
    'Decision Tree Classification': [0.99285, 0.993136, 0.992988],
    'Random Forest Classification': [0.992617, 0.993664, 0.993034]
}

# Convert to DataFrames
kc1_df = pd.DataFrame(kc1_data, index=metrics)
mw1_df = pd.DataFrame(mw1_data, index=metrics)
pc5_df = pd.DataFrame(pc5_data, index=metrics)
mc1_df = pd.DataFrame(mc1_data, index=metrics)

# Function to plot heatmap
def plot_heatmap(df, title):
    corr = df.T.corr()  # Transpose and get correlation between models
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title(title)
    plt.show()

# Plot heatmaps
plot_heatmap(kc1_df, "KC1 Dataset - Correlation Heatmap (Precision, Recall, F1 Score)")
plot_heatmap(mw1_df, "MW1 Dataset - Correlation Heatmap (Precision, Recall, F1 Score)")
plot_heatmap(pc5_df, "PC5 Dataset - Correlation Heatmap (Precision, Recall, F1 Score)")
plot_heatmap(mc1_df, "MC1 Dataset - Correlation Heatmap (Precision, Recall, F1 Score)")


# # ROC Curve & AUC

# In[45]:


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
    "Dataset 1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-KC1.csv"),
    "Dataset 2": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MW1.csv"),
    "Dataset 3": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-PC5.csv"),
    "Dataset 4": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MC1.csv")
}

plot_roc_curves(dataframes)


# # Violin Curve

# In[31]:


def plot_violin_curves(dataframes, target_column='Defective', selected_features=None):
    for name, df in dataframes.items():
        print(f"\n🔍 Plotting violin plots for {name}")
        df = df.dropna(subset=[target_column])

        if selected_features is None:
            # Automatically select a few numeric columns excluding the target
            selected_features = df.drop(columns=[target_column]).select_dtypes(include='number').columns[:3]

        for feature in selected_features:
            plt.figure(figsize=(8, 5))
            sns.violinplot(x=target_column, y=feature, data=df, palette='Set2', inner='box')
            plt.title(f"Violin Plot of '{feature}' by '{target_column}' - {name}")
            plt.xlabel('Defective')
            plt.ylabel(feature)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
dataframes = {
    "KC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-KC1.csv"),
    "MW1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MW1.csv"),
    "PC5": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-PC5.csv"),
    "MC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MC1.csv")
}

plot_violin_curves(dataframes)


# # Confusion Matrix

# In[42]:


def plot_confusion_matrices(dataframes, target_column='Defective'):
    for name, df in dataframes.items():
        print(f"\n📊 Confusion Matrix for {name}")
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Impute and scale
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.grid(False)
        plt.show()
dataframes = {
    "KC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-KC1.csv"),
    "MW1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MW1.csv"),
    "PC5": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-PC5.csv"),
    "MC1": pd.read_csv(r"E:\kiit\Project for 4th sem\Data\csv_result-MC1.csv")
}

plot_confusion_matrices(dataframes)


# In[ ]:




