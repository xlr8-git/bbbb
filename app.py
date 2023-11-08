# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:01:23 2023

@author: enrico
"""

import streamlit as st
import numpy as np
import pandas as pd
# import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# import h5py
import tensorflow as tf

from PIL import Image as PILImage

# Set page title
st.title("Soil image to particle size distribution curve app")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image
    image = PILImage.open(uploaded_image)
    # image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform image processing (you can replace this with your specific image analysis)
    # For example, let's assume you want to plot a grayscale intensity curve:
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hist, bins = np.histogram(gray_image.ravel(), bins=256, range=[0, 256])
    
    # Crop image
    # Check if the image dimensions are already 783x783
    if image.size != (400, 400):

        # Calculate the cropping coordinates to get a square image
        width, height = (400,400)
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        upper = (height - min_dim) // 2
        right = left + min_dim
        lower = upper + min_dim

        # Crop the image
        image = image.crop((left, upper, right, lower))


    # Split channels
    red_channel, green_channel, blue_channel = image.split()
    # Import channels and normalise
    red_array = np.array(red_channel)
    green_array = np.array(green_channel)
    blue_array = np.array(blue_channel)
    # Convert to dataframe
    red_df = pd.DataFrame(red_array)
    green_df = pd.DataFrame(green_array)
    blue_df = pd.DataFrame(blue_array)        
    # Convert to a list
    red_list = red_df.values.tolist()
    green_list = green_df.values.tolist()
    blue_list = blue_df.values.tolist()

    # Create input
    X = []   
    X.append([red_list,green_list,blue_list])
            
    # Scale for mobilenet
    X0 = np.array(X)
    X = tf.keras.applications.mobilenet.preprocess_input(X0)
    # Transponse data
    X = tf.transpose(X, perm=[0, 2, 3, 1])
          
    # Load your pre-trained regression model
    model = tf.keras.models.load_model('CNN.h5')  # Load your regression model
    
    # Predict parameters
    predictions = model.predict(X)
    print(predictions)
    
    # Parameter b
    predictions_b = []
    for i in predictions:
        predictions_b.append(i[0])

    # Parameter c
    predictions_c = []
    for i in predictions:
        predictions_c.append(i[1])

    # Plot the curve
    st.write("Predicted particle size distribution")
    fig = plt.figure(dpi=200,figsize=(5, 3))
    ax = fig.add_subplot(1,1,1)
    plt.grid(which='major', linewidth=0.5)
    plt.grid(which='minor', linewidth=0.25)
    plt.semilogx()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xticks([0.001,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200],labels=[0.001,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200],fontsize=6)
    plt.xlim([0.001,200])
    plt.ylim([0,1])
    plt.xlabel('Particle size $d$ (mm)')
    plt.ylabel('Finer')
    plt.plot([0.002,0.002],[0,1],c='black',linewidth=0.75)
    plt.plot([0.063,0.063],[0,1],c='black',linewidth=0.75)
    plt.plot([2,2],[0,1],c='black',linewidth=0.75)
    plt.plot([63,63],[0,1],c='black',linewidth=0.75)
    plt.text(0.0012,1.03,'Cl')
    plt.text(0.01,1.03,'Si')
    plt.text(0.27,1.03,'Sa')
    plt.text(10,1.03,'Gr')
    plt.text(100,1.03,'Bo')
    
    b = math.exp(predictions_b[0])
    c = predictions_c[0]
    
    plt.text(0.0012,0.95,'$y = 1 - \exp(-(d/b)^c)$')
    plt.text(0.0012,0.88,'b = ' + str(round(b,3)))
    plt.text(0.0012,0.81,'c = ' + str(round(c,3)))
    
    x = np.logspace(-3,3, num = 100)
    y = []
    
    # Fitted
    for j in x:
        y.append(1-math.exp(-(j/b)**c))     
    plt.plot(x,y,linewidth=1,c='tab:blue')
    
    st.pyplot(fig)

