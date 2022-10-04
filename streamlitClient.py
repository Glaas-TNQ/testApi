from fileinput import filename
from urllib import response
from tkinter import filedialog as fd
import requests
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import BytesIO, open
from PIL import Image, ImageOps
import PIL.Image
import pybase64 as base64
import cv2
import tkinter as Tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import streamlit as st


BASE="http://127.0.0.1:5000/test"


boolImg=False
boolResult=False
def TransformAndSend(file):
    arr=np.array(PIL.Image.open(file))
    _,img_encoded = cv2.imencode('.jpg', arr)
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
   # data=arr.tobytes()
    data=img_encoded.tobytes()
    response = requests.post(BASE, data, headers=headers)
    print(response)
    boolResult=True
    return response.json()


st.set_page_config(page_title="Client_Side")

#Header#

with st.container():
    st.title("Hi, I'm a not so smart ML model :wave:")
    st.subheader("         Please upload an image from your local pc")
    

with st.container():
    st.subheader("         Use the form below to upload an image")
    file = st.file_uploader(label="Upload an image in .jpg, .jpeg, .png format", type=["jpeg","png","jpg"])
    show_file=st.empty()

    if not file:
        show_file.info("Please upload a file : {}".format(''.join(['png ','jpg ','jpeg '])))
    else:
        content=file.getvalue()
    if isinstance(file, BytesIO):
        show_file.image(file)
        boolImg=True
    else:
        pass
if boolImg==True:
    txt=str(TransformAndSend(file)['prediction'])
    st.subheader("Here's my guess ---->  "+ txt)
