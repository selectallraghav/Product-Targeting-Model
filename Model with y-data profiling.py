#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport  # Import ydata-profiling
import streamlit.components.v1 as components
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load data
bm = pd.read_csv("/Users/raghav/Downloads/bank-marketing.csv", sep=";")

# Streamlit interface to input customer data
age = st.slider("Age", 18, 100, 30)
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
campaign = st.slider("Number of Contacts in Campaign", 1, 50, 1)
poutcome = st.selectbox("Outcome of Previous Campaign", ['success', 'failure', 'nonexistent', 'other'])
emp_var_rate = st.slider("Employment Variation Rate", -3.0, 3.0, 1.1)  
cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.994)
cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, 50.0, -36.4)
euribor3m = st.slider("Euribor 3 Month Rate", 0.0, 10.0, 4.857)
nr_employed = st.slider("Number of Employees", 4900, 5500, 5191)
duration = st.slider("Duration of Last Contact in Seconds", 0, 5000, 0)

# Create DataFrame from user inputs
new_customer = pd.DataFrame({
    'age': [age],
    'month': [month],
    'campaign': [campaign],
    'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed],
    'duration': [duration]
})

# --- Displaying YData Profiling Report ---
# Generate profiling report
profile = ProfileReport(bm, title="Bank Marketing Profiling Report", explorative=True)

# Save the report to a temporary HTML file
profile_html = profile.to_html()

# Display the profile report in Streamlit using HTML embedding
components.html(profile_html, height=800, scrolling=True)

# --- Prediction Model ---
# Separate features and target variable
X = bm[['age', 'month', 'campaign', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'duration']]
y = bm['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit the model with the selected columns
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=20, random_state=42))
])

# Fit on the entire training data
pipeline.fit(X_train, y_train)

# Make prediction
likelihood = pipeline.predict_proba(new_customer)[:, 1]

# Show prediction results
st.subheader("Prediction Results")
st.write(f"Likelihood of subscribing to a term deposit: {likelihood[0]:.4f}")

# Optional: Display the classification report and confusion matrix
if st.checkbox("Show Classification Report and Confusion Matrix"):
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = pipeline.predict(X_test)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)


# In[ ]:




