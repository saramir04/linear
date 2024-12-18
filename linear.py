import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    st.title("Linear Regression App")
    
    st.write("Upload a CSV file, select X and Y features, and build a linear regression model.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        
        # Feature selection
        features = data.columns.tolist()
        x_feature = st.selectbox("Select the X feature (independent variable)", features)
        y_feature = st.selectbox("Select the Y feature (dependent variable)", features)
        
        if x_feature and y_feature:
            # Prepare the data
            X = data[[x_feature]].values
            y = data[y_feature].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Display model parameters
            st.write(f"Coefficient: {model.coef_[0]}")
            st.write(f"Intercept: {model.intercept_}")
            
            # Plotting
            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, color='blue', label='Data')
            plt.plot(X, model.predict(X), color='red', label='Regression Line')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title("Linear Regression")
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
