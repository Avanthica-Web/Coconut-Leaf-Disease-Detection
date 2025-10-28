import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from ultralytics import YOLO


model = YOLO("yolov11n.yaml")
model = YOLO("yolov11n.pt")  


model.train(data="detection/data.yaml", epochs=30) 
