import numpy as np
import streamlit as st
import tensorflow as tf


def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease System For Sustainable Agriculture...")
app_mode=st.sidebar.selector('select page',['Home','Disease Recognition'])

from PIL import Image

if(app_mode=='Home'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System For Sustainable Agriculture",unsafe_allow_html=True)
    
    img= Image.open("Diseases.jpg")
    st.image(img)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    """)
#prediction page
elif(app_mode=='Disease Recognition'):
        st.header("  Plant Disease Recognition🌿🔍🍁...   ")
        test_image=st.file_uploader('Choose an Image:')
        if(st.button('Show Image')):
            st.image(test_image,width=4,use_column_width=True)
        #predict button
        if(st.button('Predict')):
            st.snow()
            st.write('Our Prediction 💡')
            result_index=model_prediction(test_image)
    
            class_name=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success('Model is predicting it {}'.format(class_name[result_index]))

