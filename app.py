import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title
st.title("🩺 Pneumonia Detection App")
st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of **Pneumonia** or is **Normal**.")

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

        # Preprocess image
        img = image.resize((150, 150))  # Match model input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        confidence_score = float(prediction[0][0])

        # Apply 80% threshold
        if confidence_score > 0.8:
            predicted_class = "Pneumonia"
            confidence = confidence_score
        else:
            predicted_class = "Normal"
            confidence = 1 - confidence_score

        # Display results
        st.subheader("🔍 Prediction Result")
        if predicted_class == "Pneumonia":
            st.error(f"⚠️ The X-ray indicates **Pneumonia** with {confidence * 100:.2f}% confidence.")
        else:
            st.success(f"✅ The X-ray appears **Normal** with {confidence * 100:.2f}% confidence.")

        # Progress bar for confidence
        st.progress(int(confidence * 100))

    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")

else:
    st.info("📁 Please upload an image to proceed.")

# Footer
st.write("---")
st.caption("Developed by Sravani Karnataka | Powered by Streamlit & TensorFlow")
