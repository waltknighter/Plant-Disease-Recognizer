import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    pred = model.predict(input_arr)
    result_index = np.argmax(pred)
    return result_index

# sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home", "Disease Recognition", "About"], label_visibility="collapsed")
st.sidebar.markdown("""
## You're using a **Deep Learning Model**
Trained on **20336** images\n
Validated on **5074** images\n
Tested on **2816** images
        
### Recognizing **99** out of **100** disease images accurately!🌿
""")

# Home Page
if(app_mode=='Home'):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition** System! 🌿
    
    This system uses advanced deep learning models to identify and diagnose diseases in **Corn**, **Grapes**, **Potatoes**, and **Tomatoes**. Simply upload a clear image of a leaf, and our model will analyze it to detect any potential disease. Our aim is to help farmers and agricultural professionals make quick and accurate decisions to protect their crops.
    
    Whether you're a farmer, researcher, or agriculture enthusiast, our platform offers you a simple, efficient, and accurate way to monitor plant health. Explore the platform and take a step towards sustainable farming!
    
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a leaf of plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience this Plant Disease Recognizer

""")
    

# About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### Dataset
    This system works on a dataset which consists of around 28k RGB images of 256x256 resolution. These images are classified into 21 classes on basis of their respective plant and diseases. These 21 classes include Corn(3 diseases), Grape(3 diseases), Potato(2 diseases), Tomato(9 diseases).
    #### Model
    Our Plant Disease Recognizer leverages a Convolutional Neural Network (CNN) architecture to accurately identify various diseases affecting crops. We trained the model on a comprehensive dataset containing thousands of labeled images of plant leaves across 21 classes, representing healthy and diseased conditions.
                
    To enhance accuracy, our model utilizes multiple layers of convolutional and pooling operations, followed by fully connected dense layers. By analyzing the features extracted from leaf images, the model makes precise predictions.
                
    Our model is designed to help in early disease detection, which is crucial for timely intervention and crop health management.
""")
    

# Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    st.markdown("""
    ### Instructions
    1. Single Leaf Focus: Ensure that only one leaf is visible and in focus in the image. Avoid including multiple leaves.
    2. Clear Background: Use a plain or minimal background to reduce noise. A light or neutral-colored background works best.
    3. Leaf Orientation: Place the leaf flat and capture it from a top-down view. Avoid angles that obscure parts of the leaf.
    4. Good Lighting: Ensure the leaf is well-lit with natural or white light. Avoid shadows or reflections on the leaf.
    5. Image Quality: Use a clear, high-resolution image. Avoid blurry or pixelated pictures.
    6. No Additional Objects: Make sure there are no other objects (e.g., hands, soil, or tools) in the frame.
    Follow these instructions for accurate results.
                
    ### Disease Recognizer
""")
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    def reset_prediction_state():
        st.session_state.prediction_done = False
    test_image = st.file_uploader("Upload image", on_change=reset_prediction_state)
    if(test_image!=None):
        # update session state with new image
        st.session_state.uploaded_image = test_image
        st.image(st.session_state.uploaded_image, use_column_width=True)
        # to hide predict button after prediction
        predict_button_placeholder = st.empty()
        # Predict Button
        if not st.session_state.prediction_done:
            if(predict_button_placeholder.button("Predict")):
                with st.spinner("Predicting..."):
                    result_index = model_prediction(st.session_state.uploaded_image)
                    # Defining Class
                    classes = ['corn(maize) having cercospora leaf spot (gray leaf spot)',
                    'corn(maize) with common rust',
                    'healthy corn(maize)',
                    'corn(maize) having northern leaf blight',
                    'grape having black rot',
                    'grape with esca (black measles)',
                    'healthy grape',
                    'grape with leaf blight (isariopsis leaf spot)',
                    'potato having early blight',
                    'healthy potato',
                    'potato having late blight',
                    'tomato having bacterial spot',
                    'tomato having early blight',
                    'healthy tomato',
                    'tomato having late blight',
                    'tomato with leaf mold',
                    'tomato with mosaic virus',
                    'tomato having septoria leaf spot',
                    'tomato having spider mites two-spotted spider mite',
                    'tomato having target spot',
                    'tomato with yellow leaf curl virus']
                    st.success("It's a {}".format(classes[result_index]))
                    st.session_state.prediction_done = True
                    predict_button_placeholder.empty()
                    # recommendation
                    recommend = ['Use resistant hybrids and avoid continuous corn planting. Employ crop rotation and manage residue to reduce fungus survival. If severe, apply fungicides like strobilurin or triazole-based products at the tasseling to early silking growth stages.',
                                'Rotate crops and use rust-resistant hybrids. Applying fungicides early in the season, when symptoms first appear, can also control the spread.',
                                '',
                                'Resistant hybrids and fungicides are effective measures. Managing crop residue through tillage and rotating with non-host crops reduces disease prevalence.',
                                'Regularly prune vines to improve air circulation and remove infected material. Fungicide applications are effective, particularly during wet weather conditions.',
                                'Avoid water stress and maintain good soil drainage. Remove and destroy infected parts of the vine. Fungicide treatment may be necessary, but prevention through proper vineyard management is key.',
                                '',
                                'Pruning to improve air circulation and using fungicides during the early stages of infection can help manage this disease.',
                                'Crop rotation and using resistant varieties are crucial. Fungicide applications should be done early in the disease cycle for best results.',
                                '',
                                'Use resistant potato varieties and fungicides as preventive measures. Avoid wet conditions around the plant base and remove infected plant debris to minimize the risk.',
                                'Implement crop rotation and use disease-free seeds. Copper-based fungicides can control bacterial spot if applied early.',
                                'Rotate crops, apply mulches to prevent soil splash, and use fungicides when symptoms appear.',
                                '',
                                'Avoid excessive moisture and use fungicides preemptively. Select resistant varieties where possible.',
                                'Improve ventilation in the growing area and avoid wet foliage by watering at the base. Fungicides may be necessary if symptoms persist.',
                                'Use certified seeds and control insect vectors like aphids. Remove infected plants to prevent the spread.',
                                'Avoid overhead irrigation and promote good air circulation. Fungicides should be applied when early signs appear.',
                                'Maintain plant moisture and use miticides if infestations are severe.',
                                'Implement crop rotation and use fungicides if the disease pressure is high.',
                                'Manage whitefly populations, the primary vector, and remove infected plants.']
                    st.write(recommend[result_index])
