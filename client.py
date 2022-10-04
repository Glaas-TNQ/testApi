from fileinput import filename
from urllib import response
from tkinter import filedialog as fd
import requests
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
from PIL import Image
import pybase64 as base64
import cv2
import tkinter as Tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import streamlit as st



BASE="http://127.0.0.1:5000/test"
root = Tk()
root.withdraw()
filename=fd.askopenfilename()
root.update()
root.destroy()
img = cv2.imread(filename)
_,img_encoded = cv2.imencode('.jpg', img)
content_type = 'image/jpeg'
headers = {'content-type': content_type}
data=img_encoded.tobytes()

# Define delta function:
def delta(classId):
    try:
        val = int(classId)
        if val==1:
            label= 'buildings'
            print("Thanks for your input, I'll be better next time :D ")
        elif val==2:
            label= 'forest'
        elif val==3:
            label= 'glacier'
        elif val==4:
            label= 'mountain'
        elif val==5:
            label= 'sea'
        elif val==6:
            label= 'street'
        elif val>6:
            print("That's not a class number, use a number between 1 and 6 ")
            check()

    except ValueError:
        print("That's not a class number!")
        check()


def check():
    response = requests.post(BASE, data, headers=headers)
    print(response.json())  
    check=input("is that correct? Type YES or NO        ")
    if(check.lower()=="yes"):
        print("Good!")
    else:
        print("What is it then? Please help me get better")
        c=input("Type:  1 for 'Buildings', 2 for 'Forest', 3 for 'Glacier', 4 for 'Mountain', 5 for 'Sea', 6 for 'Street'  ")
        delta(c)

check()


