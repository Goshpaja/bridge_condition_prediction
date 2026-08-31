import streamlit as st
import pandas as pd
import joblib
model = joblib.load(
    "bridge_condition_model.pkl"
)

preprocessor = joblib.load(
    "bridge_preprocessor.pkl"
)
st.title("🌉 Bridge Condition Prediction System")

st.write(
    "Predict whether a bridge is in Good or Poor condition "
    "based on its structural and usage characteristics."
)
age = st.number_input(
    "Age of Bridge (years)",
    min_value=0,
    max_value=200,
    value=20,
    step=1
)

traffic = st.number_input(
    "Traffic Volume",
    min_value=0,
    value=10000,
    step=100
)

material = st.selectbox(
    "Material Type",
    ["Concrete", "Steel"]
)

maintenance = st.selectbox(
    "Maintenance Level",
    [
        "No-Maintenance",
        "Annual",
        "Bi-Annual"
    ]
)

if st.button("Predict Bridge Condition"):

    input_data = pd.DataFrame({
        "Age_of_Bridge": [age],
        "Traffic_Volume": [traffic],
        "Material_Type": [material],
        "Maintenance_Level": [maintenance]
    })


    input_processed = preprocessor.transform(
        input_data
    )

   
    prediction = model.predict(
        input_processed
    )[0]

    
    probabilities = model.predict_proba(
        input_processed
    )[0]

    poor_probability = probabilities[0]
    good_probability = probabilities[1]

    
    if prediction == 1:
        st.success(
            "🌉 Bridge Condition: GOOD"
        )
    else:
        st.error(
            "⚠️ Bridge Condition: POOR"
        )

   
    st.write(
        f"Poor Condition Probability: "
        f"{poor_probability * 100:.2f}%"
    )

    st.write(
        f"Good Condition Probability: "
        f"{good_probability * 100:.2f}%"
    )