from turtle import pen
from flask import Flask, render_template, request, Response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_restful import Api, Resource
import torch
import torch.nn as nn
from torch.autograd import Variable
from werkzeug.utils import secure_filename
from torchvision.transforms import transforms
import numpy as np
import cv2
import PIL


app= Flask(__name__)
api= Api(app)
UPLOAD_FOLDER="/static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


# Model definition and reloading from best checkpoint
class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1  w=Width,Heigth, f=kernelSize , P=padding , s=stride
        #Input shape= (256,3,150,150)--->(batchSize,colorChannels,Xsize,Ysize)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Output shape (256,12,150,150)-->increased channel number
        self.bn1=nn.BatchNorm2d(num_features=12)  #n_features=channels
        #Output shape (256,12,150,150)
        self.relu1=nn.ReLU()
        #Output shape (256,12,150,150)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        #Reduce the image size by factor 2
        #Output shape (256,12,75,75)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Output shape (256,20,75,75)
        self.relu2=nn.ReLU()
        #Output shape (256,20,75,75)
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Output shape (256,32,75,75)-->increased channel number
        self.bn3=nn.BatchNorm2d(num_features=32)  #n_features=channels
        #Output shape (256,32,75,75)
        self.relu3=nn.ReLU()
        #Output shape (256,32,75,75)

        
        #The syntax we saw on the previous lines is how we can add multiple convolution layers to our model.
        #By adding more steps that gradually increase the number of channels and the accuracy fo the model
        
        self.fc=nn.Linear(in_features=32*75*75,out_features=num_classes)
        
        #Feed forward Function
        

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        #Output is going to be in matrix form(256,32,75,75), 
        #to feed the output we're going to reshape it first
            
        output=output.view(-1,32*75*75)
            
        output=self.fc(output)
            
        return output
classes=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
checkpoint=torch.load('best_checkpoint.model')
model=ConvNet(num_classes=6)
model.load_state_dict(checkpoint)
model.eval()        

# Prediction and transformer definition
transformer= transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((150,150)),
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize ([0.5,0.5,0.5], #0-1 to [-1,1] , formula(x-mean)/std
                         [0.5,0.5,0.5])
])
def prediction(img,transformer):
    
    
    
    image_tensor=transformer(img).float()
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    output=model(input)
    
    index=output.data.numpy().argmax()
    
    pred=classes[index]
    
    return pred
    
# route http posts to this method
class test(Resource):
    def post(self):
        r = request
        print("-------DEBUG-------")
        print(r.data)
        print("-------DEBUG-------")
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pred=prediction(img,transformer)
        print("Prediction =========>  "+ pred)
        result={"prediction":pred}
        return result



api.add_resource(test,"/test")  
        

if __name__=="__main__":
    app.run(debug=True)