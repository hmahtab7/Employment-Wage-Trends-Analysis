#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:50:07 2025

@author: ankitasingh
"""

# ==========================================
# Employment & Wage Trends Analysis (2022-24)
# ==========================================
# GOAL: Analyze U.S. industry employment & wage trends across NAICS sectors using logistic regression & clustering

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set plotting style for consistent visuals
plt.style.use('ggplot')
sns.set(style="whitegrid")

# ==========================================
# 1️⃣ Load the dataset
# ==========================================
# Goal: Load the cleaned BLS dataset with selected industries for 2022-24

df = pd.read_csv('/Users/ankitasingh/Desktop/DATA MINING/Project/Final/Dataset.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())     # Preview to verify structure
print(df.info())     # Check for nulls or datatype issues

missing_values = df.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

# ==========================================
# 2️⃣ Data preprocessing
# ==========================================
# Goal: Create derived metrics for analysis and clean dataset

# Create quarterly date column for time-series analysis
df['date'] = pd.to_datetime(df['year'].astype(str) + 'Q' + df['qtr'].astype(str))

# Calculate average monthly employment for each quarter
df['avg_qtrly_emplvl'] = df[['month1_emplvl', 'month2_emplvl', 'month3_emplvl']].mean(axis=1)

# Calculate employment volatility within quarter (standard deviation across 3 months)
df['emplvl_volatility'] = df[['month1_emplvl', 'month2_emplvl', 'month3_emplvl']].std(axis=1)

# Create shorter industry names for plotting
industry_name_map = {
    'All Industries': 'All Industries',
    'Insurance carriers and related activities': 'Insurance',
    'Food manufacturing': 'Food Mfg',
    'Computing infrastructure providers, data processing, web hosting, and related services': 'Computing & Hosting',
    'Telecommunications': 'Telecom',
    'Oil and gas extraction': 'Oil & Gas'
}

# Apply mapping
df['industry_short'] = df['industry_title'].replace(industry_name_map)

# ==========================================
# 2b️⃣  Data Cleaning
# ==========================================
print(f"Original dataset rows: {df.shape[0]}")

# Calculate efficiency metrics with proper handling of zero/null values
df['wage_per_employee'] = np.where(df['avg_qtrly_emplvl'] > 0, 
                                  df['total_qtrly_wages'] / df['avg_qtrly_emplvl'], 
                                  np.nan)
df['employees_per_estab'] = np.where(df['qtrly_estabs_count'] > 0, 
                                    df['avg_qtrly_emplvl'] / df['qtrly_estabs_count'], 
                                    np.nan)

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN in key calculated fields
df_clean = df.dropna(subset=['avg_qtrly_emplvl', 'wage_per_employee', 'employees_per_estab'])

# Check for extreme outliers in calculated metrics
q1_wage = df_clean['wage_per_employee'].quantile(0.25)
q3_wage = df_clean['wage_per_employee'].quantile(0.75)
iqr_wage = q3_wage - q1_wage
lower_bound_wage = q1_wage - 3 * iqr_wage  # Using 3x IQR for a more lenient threshold
upper_bound_wage = q3_wage + 3 * iqr_wage

q1_empl = df_clean['employees_per_estab'].quantile(0.25)
q3_empl = df_clean['employees_per_estab'].quantile(0.75)
iqr_empl = q3_empl - q1_empl
lower_bound_empl = q1_empl - 3 * iqr_empl
upper_bound_empl = q3_empl + 3 * iqr_empl

# Remove extreme outliers
df_clean = df_clean[(df_clean['wage_per_employee'] >= lower_bound_wage) & 
                    (df_clean['wage_per_employee'] <= upper_bound_wage) &
                    (df_clean['employees_per_estab'] >= lower_bound_empl) &
                    (df_clean['employees_per_estab'] <= upper_bound_empl)]

print(f"Cleaned dataset rows: {df_clean.shape[0]}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} rows ({(df.shape[0] - df_clean.shape[0])/df.shape[0]*100:.2f}%)")

# ==========================================
# 3️⃣ Descriptive Statistics by Industry
# ==========================================
# GOAL: Compare industry employment levels and distribution

# Filter out the 'All Industries' baseline for clearer comparison
df_filtered = df_clean[df_clean['industry_title'] != 'All Industries']

# Compute statistics by industry
industry_stats = df_filtered.groupby('industry_short').agg({
    'avg_qtrly_emplvl': ['mean', 'std', 'median'],
    'total_qtrly_wages': ['mean', 'std', 'median'],
    'avg_wkly_wage': ['mean', 'std', 'median'],
    'qtrly_estabs_count': ['mean', 'std', 'median']
})

# Flatten multi-index columns for easier access
industry_stats.columns = ['_'.join(col).strip() for col in industry_stats.columns.values]
industry_stats = industry_stats.reset_index()

# Sort by average employment
sorted_data = industry_stats.sort_values(by='avg_qtrly_emplvl_mean', ascending=False)

# Plot average employment by industry
plt.figure(figsize=(10, 6))
sns.barplot(x='industry_short', y='avg_qtrly_emplvl_mean', data=sorted_data)
plt.title('Average Quarterly Employment by Industry (2022-2024)')
plt.xlabel('Industry')
plt.ylabel('Average Employment')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# Plot average weekly wages by industry
plt.figure(figsize=(10, 6))
wage_data = industry_stats.sort_values(by='avg_wkly_wage_mean', ascending=False)
sns.barplot(x='industry_short', y='avg_wkly_wage_mean', data=wage_data)
plt.title('Average Weekly Wages by Industry (2022-2024)')
plt.xlabel('Industry')
plt.ylabel('Average Weekly Wage ($)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# 4️⃣ Correlation Analysis
# ==========================================
# Goal: Measure relationships among core metrics across industries

# Select relevant columns for correlation
selected_cols = ['avg_qtrly_emplvl', 'total_qtrly_wages', 'avg_wkly_wage', 
                 'qtrly_estabs_count', 'wage_per_employee', 'emplvl_volatility']

# Calculate overall correlation matrix
correlation_matrix = df_clean[selected_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# ==========================================
# 5️⃣ Time Series Trend Analysis
# ==========================================
# Goal: Analyze employment and wage trends over time (2022-2024)

# Group by industry and date for time series
time_data = df_clean.groupby(['industry_title', 'date']).agg({
    'avg_qtrly_emplvl': 'mean',
    'avg_wkly_wage': 'mean'
}).reset_index()

# Filter out All Industries for clearer visualization
time_data = time_data[time_data['industry_title'] != 'All Industries']

# Employment trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=time_data, x='date', y='avg_qtrly_emplvl', 
             hue='industry_title', marker='o')
plt.title('Employment Trends by Industry (2022-2024)')
plt.xlabel('Date')
plt.ylabel('Average Quarterly Employment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Wage trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=time_data, x='date', y='avg_wkly_wage', 
             hue='industry_title', marker='o')
plt.title('Average Weekly Wage Trends by Industry (2022-2024)')
plt.xlabel('Date')
plt.ylabel('Average Weekly Wage ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 6️⃣ Logistic Regression
# ==========================================
# Goal: Classify high vs low wage industries and identify key predictors

print("\n--- Logistic Regression Analysis ---")

# Drop text columns
df_clean_reg = df_clean.drop(columns=['industry_title', 'area_title', 'industry_short', 'date'])

# Ensure industry_code is treated as categorical 
if 'industry_code' in df_clean_reg.columns:
    df_clean_reg['industry_code'] = df_clean_reg['industry_code'].astype(str)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_clean_reg, columns=['industry_code'], drop_first=True)

# Create binary target variable based on median wage
median_wage = df_encoded['total_qtrly_wages'].median()
df_encoded['high_wage'] = (df_encoded['total_qtrly_wages'] > median_wage).astype(int)

# Select features and target
X = df_encoded.drop(columns=['total_qtrly_wages', 'high_wage'])
X = X.select_dtypes(include=[np.number])  # Keep only numeric features
y = df_encoded['high_wage']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate model
print("\nClassification Report:")
classification_report_output = classification_report(y_test, y_pred)
print(classification_report_output)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Logistic Regression")
plt.grid(False)
plt.tight_layout()
plt.show()

# Feature Importance
feature_importance = pd.Series(np.abs(log_reg.coef_[0]), index=X.columns).sort_values(ascending=False)
top_features = feature_importance.head(10)
print("\nTop 10 Important Features for Predicting High Wages:")
print(top_features)

# Plot feature importance
plt.figure(figsize=(12, 6))
top_features.plot(kind='bar')
plt.title('Top 10 Features for Predicting High-Wage Industries')
plt.xlabel('Features')
plt.ylabel('Coefficient Magnitude')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7️⃣ Clustering Analysis
# ==========================================
# Goal: Group similar industries based on employment and wage patterns
print("\n--- K-Means Clustering Analysis ---")

# Select variables for clustering
cluster_vars = ['avg_qtrly_emplvl', 'avg_wkly_wage', 'qtrly_estabs_count', 
                'emplvl_volatility', 'wage_per_employee', 'employees_per_estab']

# Prepare data
df_cluster = df_clean[cluster_vars + ['industry_title', 'industry_short']].dropna()

# Scale features
X_cluster = df_cluster[cluster_vars]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.show()

# Apply K-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Plot clusters using original variables
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='avg_qtrly_emplvl', y='avg_wkly_wage', 
               hue='cluster', palette='Set2', s=80)
plt.title('KMeans Clustering of Industries')
plt.xlabel('Average Quarterly Employment')
plt.ylabel('Average Weekly Wage')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Cluster characteristics
cluster_profile = df_cluster.groupby('cluster').agg({
    'avg_qtrly_emplvl': 'mean',
    'avg_wkly_wage': 'mean',
    'qtrly_estabs_count': 'mean',
    'emplvl_volatility': 'mean',
    'wage_per_employee': 'mean',
    'employees_per_estab': 'mean'
}).reset_index()
print("\nCluster Profiles:")
print(cluster_profile)

# Map industries to clusters
industry_clusters = df_cluster.groupby(['industry_title', 'cluster']).size().reset_index(name='count')
print("\nIndustry Cluster Assignments:")
print(industry_clusters)