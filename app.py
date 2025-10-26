#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install tensorflow matplotlib numpy


# In[3]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image


# In[5]:


train_dir = "C:/Users/DELL/Desktop/chest_xray/train"
val_dir   = "C:/Users/DELL/Desktop/chest_xray/val"
test_dir  = "C:/Users/DELL/Desktop/chest_xray/test"


# In[7]:


# Training data
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)

# Validation data
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)

# Test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Input(shape=(150, 150, 3)),   # 👈 instead of putting input_shape in Conv2D
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[11]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_data = image_dataset_from_directory(
    "C:/Users/DELL/Desktop/chest_xray/train",
    image_size=(150,150),
    batch_size=32
)

val_data = image_dataset_from_directory(
    "C:/Users/DELL/Desktop/chest_xray/val",
    image_size=(150,150),
    batch_size=32
)

test_data = image_dataset_from_directory(
    "C:/Users/DELL/Desktop/chest_xray/test",
    image_size=(150,150),
    batch_size=32
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

# ----------------------------
# 1️⃣ Evaluate CNN on test data
# ----------------------------
loss, acc = model.evaluate(test_data)
print(f"Test Loss: {loss:.2f}, Test Accuracy: {acc:.2f}")

# ----------------------------
# 2️⃣ Plot Accuracy
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 3️⃣ Plot Loss
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Model Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 4️⃣ Confusion Matrix
# ----------------------------
# Collect predictions
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images)
    preds = (preds > 0.5).astype("int32").flatten()  # binary classification
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"])
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ----------------------------
# 5️⃣ ROC Curve (Optional)
# ----------------------------
y_prob = []
for images, _ in test_data:
    preds = model.predict(images)
    y_prob.extend(preds.flatten())

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')  # diagonal line
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ----------------------------
# 6️⃣ Precision-Recall Curve (Optional)
# ----------------------------
precision, recall, _ = precision_recall_curve(y_true, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import numpy as np

def plot_model_accuracy_pie(models, accuracies,
                            sort=True,
                            explode_best=True,
                            colors=None,
                            title="Comparison of Models (Accuracy-based)",
                            save_path=None,
                            figsize=(7,7)):
    """
    models: list of model names
    accuracies: list/array of accuracies (decimals 0-1 or percentages 0-100)
    sort: if True, sorts by accuracy descending
    explode_best: if True, "explode" the best model slice
    colors: list of colors (same length as models) or None for default palette
    save_path: filepath to save the figure (e.g. 'comparison.png') or None
    """
    models = list(models)
    accuracies = np.array(accuracies, dtype=float)

    # Convert decimals (<=1) to percentages
    if np.all(accuracies <= 1.0):
        accuracies = accuracies * 100.0

    # Sort descending so largest slice appears first (nice visual order)
    if sort:
        idx = np.argsort(accuracies)[::-1]
        accuracies = accuracies[idx]
        models = [models[i] for i in idx]

    # Explode the best slice a bit
    explode = [0.12 if (i == 0 and explode_best) else 0.0 for i in range(len(models))]

    # Default colors if none provided
    if colors is None:
        # a pleasant pastel palette; change if you like
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ffb3e6', '#c2c2f0', '#f0b3ff']
    # trim/extend colors to match length
    if len(colors) < len(models):
        # repeat colors if not enough
        colors = (colors * (len(models)//len(colors) + 1))[:len(models)]
    else:
        colors = colors[:len(models)]

    plt.figure(figsize=figsize)
    wedges, texts, autotexts = plt.pie(
        accuracies,
        labels=None,                # we'll use legend to keep chart clean
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.6,            # position of percent text
        labeldistance=1.05,
        explode=explode,
        colors=colors,
        wedgeprops={'edgecolor':'black', 'linewidth':0.5}
    )

    # Improve autopct font size and weight
    for t in autotexts:
        t.set_fontsize(10)
        t.set_weight('bold')

    plt.title(title, fontsize=14)
    plt.axis('equal')  # Equal aspect ensures pie is drawn as a circle

    # Place legend to the left (or change bbox_to_anchor to move it)
    plt.legend(wedges, models, title="Models", loc="center left",
               bbox_to_anchor=(1.02, 0.5), fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# -------------------------
# Example usage
# -------------------------
models = ["Logistic Regression", "Random Forest", "Decision Tree", "Naive Bayes", "CNN"]
# You can pass decimals (0-1) or percentages (0-100). Both work.
accuracies = [0.208, 0.208, 0.18, 0.202, 0.201]  # example (decimals)
plot_model_accuracy_pie(models, accuracies, save_path="model_accuracy_pie.png")


# In[17]:


# Replace with your test image path
img_path = "C:/Users/DELL/Desktop/Python/download (1).jpeg"

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: Pneumonia")
else:
    print("Prediction: Normal")

# Optional: show image with prediction
plt.imshow(img)
plt.title("Prediction: Pneumonia" if prediction[0][0] > 0.5 else "Prediction: Normal")
plt.axis('off')
plt.show()


# In[19]:


# Replace with your test image path
img_path = "C:/Users/DELL/Desktop/Python/download.jpeg"

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: Pneumonia")
else:
    print("Prediction: Normal")

# Optional: show image with prediction
plt.imshow(img)
plt.title("Prediction: Pneumonia" if prediction[0][0] > 0.5 else "Prediction: Normal")
plt.axis('off')
plt.show()


# In[21]:


# Replace with your test image path
img_path = "C:/Users/DELL/Desktop/Python/download (2).jpeg"

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: Pneumonia")
else:
    print("Prediction: Normal")

# Optional: show image with prediction
plt.imshow(img)
plt.title("Prediction: Pneumonia" if prediction[0][0] > 0.5 else "Prediction: Normal")
plt.axis('off')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# Example results (replace these with your actual values from evaluation)
models = ["Logistic Regression", "Random Forest", "Decision Tree", "Naive Bayes", "CNN"]

accuracy =  [0.64, 0.60, 0.53, 0.62, 0.92]
precision = [0.46, 0.52, 0.53, 0.48, 0.92]
recall =    [0.64, 0.60, 0.53, 0.62, 0.94]
f1 =        [0.51, 0.53, 0.53, 0.53, 0.93]

# Grouped bar chart
x = np.arange(len(models))  # model positions
width = 0.2  # bar width

plt.figure(figsize=(10,6))

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
plt.bar(x - 0.5*width, precision, width, label='Precision')
plt.bar(x + 0.5*width, recall, width, label='Recall')
plt.bar(x + 1.5*width, f1, width, label='F1-score')

# Labels and formatting
plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Model Performance Comparison Including CNN")
plt.legend()
plt.tight_layout()
plt.show()


# In[25]:


X_train = []
y_train = []

for images, labels in train_data:
    flattened_images = images.numpy().reshape(images.shape[0], -1)
    X_train.extend(flattened_images)
    y_train.extend(labels.numpy())

X_train = np.array(X_train)
y_train = np.array(y_train)


# In[31]:


model.save('pneumonia_model.keras')


# In[35]:


import warnings
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

from tensorflow.keras.models import load_model
model = load_model('pneumonia_model.keras')


# In[37]:


from tensorflow.keras.models import load_model

# Load your model
model = load_model('pneumonia_model.keras')  # or 'pneumonia_model.h5'


# In[39]:


import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and let the model predict whether it’s Pneumonia or Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_column_width=True)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

    st.success(f"Prediction: {result}")


# In[ ]:




