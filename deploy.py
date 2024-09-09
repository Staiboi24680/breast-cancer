# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Load your trained model (ensure you have the correct path to the model)
# model = tf.keras.models.load_model(r"C:\Users\STAIBOI\Desktop\New folder (2)\Dataset_BUSI_with_GT\breast_cancer_classifier.h5")

# # Set up the app title
# st.title('Breast Cancer Classification App')

# # Allow the user to upload an image
# uploaded_file = st.file_uploader("Upload a breast cancer image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Preprocess the image for the model
#     img = image.resize((224, 224))  # Resize to the required input size
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
#     # Make a prediction
#     prediction = model.predict(img_array)
#     class_names = ['Benign', 'Malignant', 'Normal', 'Unknown']  # Ensure these match your model's classes
#     predicted_class = class_names[np.argmax(prediction)]
    
#     # Show prediction result
#     st.write(f"Prediction: {predicted_class}")






import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\STAIBOI\Desktop\New folder (2)\Dataset_BUSI_with_GT\breast_cancer_classifier.h5")

# Set up the app title
st.title('Breast Cancer Classification App')

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload a breast cancer image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to RGB (in case it's not already in RGB)
    image = image.convert("RGB")
    
    # Resize the image for the model
    img = image.resize((224, 224))  # Resize to the required input size
    
    # Convert the image to a numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    
    # Check the shape of the img_array
    st.write(f"Shape of image array before expansion: {img_array.shape}")
    
    # Add a batch dimension (so the shape becomes (1, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Check the shape of the img_array after expansion
    st.write(f"Shape of image array after expansion: {img_array.shape}")
    
    # Make a prediction
    try:
        prediction = model.predict(img_array)
        class_names = ['Benign', 'Malignant', 'Normal', 'unknown'] 
        predicted_class = class_names[np.argmax(prediction)]
        
        # Show prediction result
        st.write(f"Prediction: {predicted_class}")
    except Exception as e:
        st.write(f"Error making prediction: {e}")
