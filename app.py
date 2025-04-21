import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, 
                            classification_report, roc_curve, precision_recall_curve)

# Configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# 1. Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\vasanth\Downloads\Telegram Desktop\Fraud_Analysis_Dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# 2. Data Preprocessing
def preprocess_data(df):
    df = df.copy()
    # Drop non-numeric columns
    df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)
    # Encode categorical columns
    df["type"] = LabelEncoder().fit_transform(df["type"])
    # Separate features and target
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Classifier options
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Training Functionality
def train_model():
    st.title("Fraud Detection Model Trainer")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    X, y, scaler = preprocess_data(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Sidebar - model selection
    st.sidebar.header("Model Configuration")
    classifier_name = st.sidebar.selectbox(
        "Select Classifier", 
        list(CLASSIFIERS.keys())
    )
    
    # Sidebar - hyperparameters
    params = {}
    if classifier_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of trees", 50, 500, 100)
        params["max_depth"] = st.sidebar.slider("Max depth", 2, 20, 5)
    elif classifier_name == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("Number of estimators", 50, 500, 100)
        params["learning_rate"] = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1)

    # Sidebar - financial assumptions using sliders
    st.sidebar.header("Financial Assumptions")
    TP_REWARD = st.sidebar.slider("Reward for detecting fraud (TP)", min_value=100, max_value=1000, value=487, step=1)
    FP_COST = st.sidebar.slider("Cost of false positive (FP)", min_value=0, max_value=200, value=44, step=1)
    FN_LOSS = st.sidebar.slider("Loss from missed fraud (FN)", min_value=100, max_value=2000, value=500, step=10)
    
    # Training trigger
    if st.sidebar.button("Train Model"):
        try:
            with st.spinner(f"Training {classifier_name}..."):
                # Get classifier with params
                clf = CLASSIFIERS[classifier_name]
                clf.set_params(**params)
                
                # Train model
                clf.fit(X_train, y_train)
                
                # Save model
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", f"{classifier_name.replace(' ', '_').lower()}_model.pkl")
                joblib.dump((clf, scaler), model_path)
                
                # Predictions
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
                
                # Metrics
                st.success("Training Complete!")
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                    st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
                with col2:
                    st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                
                # Financial impact calculation
                st.subheader("Financial Impact")
                tn, fp, fn, tp = cm.ravel()
                expected_revenue = tp * TP_REWARD
                expected_loss = fp * FP_COST + fn * FN_LOSS
                net_profit = expected_revenue - expected_loss
                st.metric("Expected Revenue (Saved from Fraud)", f"${expected_revenue:,.2f}")
                st.metric("Expected Loss (Costs & Missed Fraud)", f"${expected_loss:,.2f}")
                st.metric("Net Profit", f"${net_profit:,.2f}")
                
                # ROC Curve
                if y_prob is not None:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label='ROC curve')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    st.pyplot(fig)
                    st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.2f}")
                
                # Feature Importance
                if hasattr(clf, "feature_importances_"):
                    st.subheader("Feature Importance")
                    importance = clf.feature_importances_
                    features = df.drop("isFraud", axis=1).columns
                    fig, ax = plt.subplots()
                    ax.barh(features, importance)
                    ax.set_xlabel('Importance Score')
                    st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

# Load available models
def load_models():
    models = {}
    model_dir = "models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                model_name = file.replace("_model.pkl", "").replace("_", " ").title()
                model_path = os.path.join(model_dir, file)
                models[model_name] = joblib.load(model_path)
    return models

# Function to preprocess input data
def preprocess_input(input_data, scaler):
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode type
    type_mapping = {
        'PAYMENT': 0,
        'TRANSFER': 1,
        'CASH_OUT': 2,
        'DEBIT': 3,
        'CASH_IN': 4
    }
    df['type'] = df['type'].map(type_mapping)
    
    # Scale features
    X_scaled = scaler.transform(df)
    return X_scaled

# Prediction Functionality
def predict_fraud():
    st.title("Fraud Detection System")
    st.write("""
    This application uses machine learning models to detect fraudulent transactions.
    """)
    
    # Load models
    models = load_models()
    if not models:
        st.warning("No trained models found. Please train models first in the Training section.")
        return
    
    # Model selection
    model_name = st.selectbox("Select Model", list(models.keys()))
    model, scaler = models[model_name]
    
    # Input form
    st.sidebar.header("Transaction Details")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step (hour)", min_value=0, max_value=744, value=0)
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=1000.0)
            newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=0.0)
            
        with col2:
            transaction_type = st.selectbox(
                "Transaction Type",
                ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
            )
            oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0)
            newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1000.0)
            isFlaggedFraud = st.selectbox("Is Flagged Fraud", [0, 1])
        
        submitted = st.form_submit_button("Check for Fraud")
    
    if submitted:
        # Prepare input data
        input_data = {
            'step': step,
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'isFlaggedFraud': isFlaggedFraud
        }
        
        # Preprocess input
        try:
            X = preprocess_input(input_data, scaler)
            
            # Make prediction
            prediction = model.predict(X)
            probability = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
            
            # Display results
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")
            
            if probability is not None:
                st.write(f"Fraud Probability: {probability:.2%}")
                
                # Visualize probability
                fig, ax = plt.subplots()
                ax.barh(['Fraud Probability'], [probability], color='red' if prediction[0] == 1 else 'green')
                ax.set_xlim(0, 1)
                ax.set_title('Transaction Risk Assessment')
                st.pyplot(fig)
            
            # Explanation
            st.subheader("Explanation")
            if prediction[0] == 1:
                st.write("""
                This transaction has been flagged as potentially fraudulent based on the model's analysis.
                Please review this transaction carefully and consider additional verification steps.
                """)
                st.write("**Recommended Actions:**")
                st.write("- Place a temporary hold on the transaction")
                st.write("- Contact the account holder for verification")
                st.write("- Review similar recent transactions")
            else:
                st.write("""
                This transaction appears to be legitimate based on the model's analysis.
                However, please continue to monitor for any suspicious activity.
                """)
                
        except Exception as e:
            st.error(f"Error processing transaction: {str(e)}")

# Main App
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Fraud Prediction", "Model Training"])
    
    if app_mode == "Fraud Prediction":
        predict_fraud()
    else:
        train_model()

if __name__ == "__main__":
    main()