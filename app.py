import streamlit as st
import torch
import detect
from PIL import Image, ImageOps
from io import *
import glob
from datetime import datetime
import os
#import wget
import time




#def imageInput(device, src):
def imageInput(src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            img_orig = ImageOps.exif_transpose(img)
            with col1:
                st.image(img_orig, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            #-imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            #-outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            #-with open(imgpath, mode="wb") as f:
            #-    f.write(image_file.getbuffer())

            ### call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/cons0205/weights/best.pt', force_reload=True)
            # model.cuda() if device == 'cuda' else model.cpu()
            #-pred = model(imgpath)
            pred = model(img_orig)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                # not saving it to git
                # im_base64.save(outputpath)

            # --Display predicton

            img_ =Image.open(im_base64)
            # img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.':
        # Image selector slider
        test_images = os.listdir('data/images/')
        test_image = st.selectbox('Please select a test image:', test_images)
        image_file = 'data/images/' + test_image
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            # solve image rotations problem
            img_orig = ImageOps.exif_transpose(img)
            st.image(img_orig, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/cons0205/weights/best.pt', force_reload=True)
                # model.cuda() if device == 'cuda' else model.cpu()
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                    # --Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])

    st.header('🚧Construction Object Detection Model')
    st.subheader('👈🏽Select the options')

    #imageInput(deviceoption, datasrc)
    imageInput(datasrc)

if __name__ == '__main__':
    main()