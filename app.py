import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('predict_salary.pkl')
scaler = joblib.load('scaler.pkl')

# Page configuration
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            background: linear-gradient(to right, #36d1dc, #5b86e5);
            -webkit-background-clip: text;
            color: transparent;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
            text-align: center;
        }
        .salary {
            font-size: 30px;
            font-weight: bold;
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">💼 Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">📊 Estimate your salary based on your experience level</div>', unsafe_allow_html=True)

# Experience Input
st.markdown("### 🧑‍💻 Select Your Experience:")
years = list(range(0, 21))
years_exp = st.selectbox("🔢 Years of Experience", years)

# Predict Button
if st.button("✨ Predict Salary"):
    input_data = np.array([[years_exp]])
    input_scaled = scaler.transform(input_data)
    predicted_salary = model.predict(input_scaled)

    # Display in styled box
    st.markdown(f"""
        <div class="box">
            <h4>Predicted Salary for {years_exp} years of experience is:</h4>
            <p class="salary">₹{predicted_salary[0]:,.2f}</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center><small style='color:gray'>🚀 Made using Streamlit</small></center>", unsafe_allow_html=True)
