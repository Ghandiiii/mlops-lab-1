import streamlit as st
import requests

st.set_page_config(
    page_title="Food-11 Classifier",
    page_icon="🍽️",
)

st.title("Food-11 Image Classifier")
st.write("Upload a food image to classify it using our trained ResNet18 model.")

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=350)

    if st.button("Predict"):
        with st.spinner("Classifying image..."):
            try:
                response = requests.post(
                    "http://inference:8000/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=60,
                )

                response.raise_for_status()
                result = response.json()

                st.success(
                    f"Prediction: {result['predicted_class']}"
                )

                st.write(
                    f"Confidence: {result['confidence']:.2%}"
                )

            except requests.exceptions.RequestException as error:
                st.error(f"Prediction request failed: {error}")
