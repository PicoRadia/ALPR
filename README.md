# FLP
FLP which stands for Find License Plate is an Automatic License Plate recognition project realised by Radia EL HAMDOUNI and Lloyd Mace.

 + The detection part uses state of the art YOLOV5 trainde on 10000 images to detect the license plate.
 + The segmentation is used with classic image processing algorithms using OpenCV.
 + The OCR (Optical Character Recognition) is done using the tool tessract.

![alt text](https://github.com/PicoRadia/FLP/blob/main/flp1.png)
This Flask app developped for ease of use of FLP let's users upload their own pictures of cars with license plate .
![alt text](https://github.com/PicoRadia/FLP/blob/main/flp4.png)
<br>
The result is then shown after model inference.
