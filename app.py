import streamlit as st
import pandas as pd
import argparse
import joblib
import os

from PIL import Image


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser.parse_args()

    def add_core_args(self):
        self.parser = argparse.ArgumentParser(description='DEVICE Captioner')
        self.parser.add_argument(
            "--save_dir", type=str, default="save", help="Where to save the model"
        )
        self.parser.add_argument(
            "--device", type=str, default="cpu", help="Set device"
        )


class App():
    def __init__(self, args):
        self.save_dir = args.save_dir
        self.model_path = os.path.join(self.save_dir, "rf_model.pkl")
        self.cm_path = os.path.join(self.save_dir, "confusion_matrix.png")
        self.load_model()
        self.load_confusion_matrix()
        
    def load_model(self):
        self.model = joblib.load(self.model_path)

    def load_confusion_matrix(self):
        self.cm_image = Image.open(self.cm_path)
        
        
    def show_prediction_page(self):
        st.header("📊 Make Predictions")
        st.write("Enter patient features to predict adherence:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Features")
            
            # Input fields for features
            age = st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=50,
                step=1,
                help="Patient's age in years"
            )
            
            annual_claim = st.number_input(
                "Annual Claim Amount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=5000.0,
                step=100.0,
                help="Total annual claim amount"
            )
            
            units_total = st.number_input(
                "Units Total",
                min_value=0,
                max_value=10000,
                value=100,
                step=1,
                help="Total number of units"
            )
            
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            
            if predict_button:
                if self.model is not None:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'AGE': [age],
                        'ANNUALCLAIMAMOUNT': [annual_claim],
                        'UNITSTOTAL': [units_total]
                    })
                    
                    # Make prediction
                    prediction = self.model.predict(input_data)[0]
                    prediction_proba = self.model.predict_proba(input_data)[0]
                    
                    # Display results
                    st.markdown("### Prediction:")
                    if prediction == 1:
                        st.success("✅ **ADHERENT**")
                        confidence = prediction_proba[1] * 100
                    else:
                        st.error("❌ **NON-ADHERENT**")
                        confidence = prediction_proba[0] * 100
                    
                    st.markdown(f"### Confidence: **{confidence:.2f}%**")
                    
                    # Show probability distribution
                    st.markdown("### Probability Distribution:")
                    prob_df = pd.DataFrame({
                        'Class': ['Non-Adherent', 'Adherent'],
                        'Probability': [prediction_proba[0], prediction_proba[1]]
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                    
                    # Show input summary
                    st.markdown("### Input Summary:")
                    st.dataframe(input_data, use_container_width=True)
                else:
                    st.error("Model not loaded. Please check if the model file exists.")

    def show_performance_page(self):
        st.header("📈 Model Performance")
        
        # Display confusion matrix
        if self.cm_image is not None:
            st.subheader("Confusion Matrix")
            st.image(self.cm_image, use_column_width=True)
            
            st.markdown("""
            ### Model Metrics
            The confusion matrix shows the performance of the Random Forest classifier:
            - **True Negatives (TN)**: Correctly predicted Non-Adherent patients
            - **False Positives (FP)**: Incorrectly predicted as Adherent
            - **False Negatives (FN)**: Incorrectly predicted as Non-Adherent
            - **True Positives (TP)**: Correctly predicted Adherent patients
            """)
        else:
            st.warning("Confusion matrix image not found. Please run the model training script first.")
        
        # Model information
        st.markdown("---")
        st.subheader("Model Information")
        st.markdown("""
            **Model Type:** Random Forest Classifier
            
            **Features Used:**
            - Age (years)
            - Annual Claim Amount ($)
            - Units Total
            
            **Hyperparameters:**
            - Max Depth: 7
            - Number of Estimators: 200
            - Criterion: Entropy
            - Random State: 1
        """)
        
        
    def UI(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page:", ["Model Prediction", "Model Performance"])
        
        if page == "Model Prediction":
            self.show_prediction_page()
        elif page == "Model Performance":
            self.show_performance_page()


if __name__=="__main__":
    flags = Flags()
    args = flags.get_parser()
    app = App(args)
    app.UI()
