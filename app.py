from flask import Flask,render_template, request ,jsonify
import torch
import cv2
import torch
from PIL import Image
import numpy as np
import pytesseract
from PIL import Image


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

#### Utility functions
def get_bbox(preds):
    ''' This function takes as an input the result of the yolov5 prediction and 
    returns a list contaning the coordinates of the bbox => #x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels) '''
    if len(preds) == 1 :
        bbox = np.array(preds)[0][0:4]
        return bbox
    else :
        vals = preds[0]
        bbox = np.array(vals)[0:4]
        return bbox 


# A function to crop the license Plate
def crop_image(img , bbox):
    ''' This funtion crops the image giving the bounding box to leave only the license plate '''
    img = cv2.imread(img)
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])
    crop_img = img[y:y+h,x:x+w]
    return crop_img


# Resize the image
def resize_image(img):
    ''' This function blaa bla '''
    img_shape = list(img.shape)
    resize_img = cv2.resize(img  , (img_shape[1]*3, img_shape[0]*3))
    return resize_img

### # # 


@app.route('/', methods=['GET' , 'POST'])
def success():
    if request.method == 'POST':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='./best.pt')  # custom model
        #path_or_model='./last.pt', map_location='cpu'  )  # custom model
        f = request.files['file']
        saveLocation = f.filename
        f.save(saveLocation)
        result= model(saveLocation)
        result_list = result.xyxy[0] 
        bbox = get_bbox(result_list)
        license_plate = crop_image(saveLocation,bbox)
        new_img = resize_image(license_plate)
        # Gaussian Blur
        # Gray Scale
        grayscale_resize_test_license_plate = cv2.cvtColor( 
        new_img, cv2.COLOR_BGR2GRAY) 
        gaussian_blur_license_plate = cv2.GaussianBlur( 
        grayscale_resize_test_license_plate, (5, 5), 0)

        prediction = pytesseract.image_to_string(gaussian_blur_license_plate, lang ='eng', 
        config ='--oem 3 -l eng --oem 1 --psm 11 tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
        prediction = "".join(prediction.split()).replace(":", "").replace("-", "") 
        if prediction == '':
            result =  {'result' : "Can't detect License Plate"}
            return render_template('error.html' , value = "Can't detect License Plate")

        else:
            result = {'result' : prediction}

        return render_template('success.html' , value = prediction)

    return render_template('index.html')        
        
    
        

if __name__ == '__main__':
    app.run(debug=True)
