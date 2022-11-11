try:
 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import tensorflow as tf
    from pathlib import Path
    import base64


    import os
    import json

    import requests  # pip install requests
    import streamlit as st  # pip install streamlit
    from streamlit_lottie import st_lottie  # pip install streamlit-lottie



    from glob import glob
    from PIL import Image as pil_image
    from matplotlib.pyplot import imshow, imsave
    from IPython.display import Image as Image

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from keras.utils.np_utils import to_categorical
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from keras.layers import Concatenate, GlobalAveragePooling2D
    from keras.optimizers import Adam, SGD
    from keras import regularizers, initializers
    from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
    from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
    from keras.layers.merge import concatenate
    from keras.preprocessing.image import ImageDataGenerator
    #from keras.callbacks import *
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.utils import plot_model
    from keras.models import load_model
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.nasnet import NASNetLarge
    from keras.applications.resnet50 import ResNet50
    from keras.applications.vgg16 import VGG16
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
    
   
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_B19OOa.json")
    # Use local CSS
    def local_css(file_name):
         with open(file_name) as f:
              st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style/style.css")          
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'unk':'unknown',
    'lc':'leshimone cutane'
   
    }


    lesion_classes_dict = {
    0:'unk',
    1:'nv',
    2:'mel',
    3:'lc'
    
   
    }

    
    
    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded    
    
    model=tf.keras.models.load_model('training_1/resnetModelTake3')
except Exception as e:
    print(e)
 
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
 
 
class FileUpload(object):
 
    def __init__(self):
        self.fileTypes = ["csv", "png", "jpg"]
 
    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        
        #st.info(__doc__)
        
        header_html = """
    <nav class="navbar  fixed-top  navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
       
      <img src='data:image/png;base64,{}' class='img-fluid' style="max-width: 200px; max-height: 65px">
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item active">
            <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#https://youtube.com/dataprofessor" target="_blank">About</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#https://twitter.com/thedataprof" target="_blank">Contact</a>
          </li>
        </ul>
      </div>
    </nav>
    """.format(
            img_to_bytes("C:/Users/ASUS/gou.png")


        )
        
        st.markdown(
            header_html, unsafe_allow_html=True,
        )
        with st.container():
            
            st.title("Project Goal")
            st.write(
                "Assist passions in determining the sort of lesion they are dealing with."
            )
        st.write("[Learn More >](https://pythonandvba.com)")
        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.header("Drop your scan here !")
                st.markdown(STYLE, unsafe_allow_html=True)
                file = st.file_uploader("Upload file", type=self.fileTypes)
                show_file = st.empty()
                if not file:
                    show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
                    return
                content = file.getvalue()
        
                if isinstance(file, BytesIO):
                    show_file.image(file)
                    resized_image2 = np.asarray(pil_image.open(file).resize((120,120)))
                    image_array2 = np.asarray(resized_image2.tolist())
                    test_image2 = image_array2.reshape(1,120,120,3)
                    print('ok')
                    prediction_class = model.predict(test_image2)
                    prediction_class = np.argmax(prediction_class,axis=1)
            
                    st.title(lesion_type_dict[lesion_classes_dict[prediction_class[0]]])


                else:
                    data = pd.read_csv(file)
                    st.dataframe(data.head(10))
                file.close()
        
        with right_column:
            st_lottie(lottie_hello, height=300, key="coding")

        
 
 
if __name__ ==  "__main__":
    
   
    helper = FileUpload()
    helper.run()