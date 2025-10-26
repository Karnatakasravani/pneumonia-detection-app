#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Pneumonia Detection App
# Streamlit-Optimized Version
# -----------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# -----------------------------
# Load the trained model
# -----------------------------
model = tf.keras.models.load_model("pneumonia_model.keras")

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Pneumonia Detection App", page_icon="🫁", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image, and the model will predict whether it's **Normal** or **Pneumonia**.")

# -----------------------------
# Upload image section
# -----------------------------
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else (1 - prediction[0][0])

    # Display result
    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")

    if predicted_class == "Pneumonia":
        st.error("⚠️ The model predicts this X-ray shows signs of pneumonia. Please consult a doctor.")
    else:
        st.success("✅ The model predicts this X-ray is normal.")

# -----------------------------
# Optional: Model Performance Plots
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Performance Plots")

# Example: replace with actual history object if available
try:
    history  # if history exists
except NameError:
    st.info("Training history not available. Skipping performance plots.")
else:
    # Accuracy Plot
    fig_acc, ax_acc = plt.subplots(figsize=(6,4))
    ax_acc.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    ax_acc.set_title("Model Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)
    st.pyplot(fig_acc)

    # Loss Plot
    fig_loss, ax_loss = plt.subplots(figsize=(6,4))
    ax_loss.plot(history.history['loss'], label='Train Loss', marker='o')
    ax_loss.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    ax_loss.set_title("Model Loss over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)
    st.pyplot(fig_loss)

# -----------------------------
# Optional: Confusion Matrix, ROC, Precision-Recall
# -----------------------------
try:
    test_data  # if test_data exists
except NameError:
    st.info("Test data not available. Skipping confusion matrix and ROC/PR plots.")
else:
    # Collect predictions
    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_data:
        preds = model.predict(images)
        preds_binary = (preds > 0.5).astype("int32").flatten()
        y_pred.extend(preds_binary)
        y_true.extend(labels.numpy())
        y_prob.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"],
                ax=ax_cm)
    ax_cm.set_title("CNN Confusion Matrix")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    st.pyplot(fig_cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
    ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax_roc.plot([0,1], [0,1], 'k--')
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)
    st.pyplot(fig_roc)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig_pr, ax_pr = plt.subplots(figsize=(6,4))
    ax_pr.plot(recall, precision, label="Precision-Recall Curve")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid(True)
    st.pyplot(fig_pr)

# Footer
st.markdown("---")
st.caption("Developed by Sravani Karnataka | Pneumonia Detection App using Deep Learning 🫁")
