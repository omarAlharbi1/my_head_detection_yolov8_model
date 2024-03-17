# my_head_detection_yolov8_model
you can find the model in the releases section
this model does use yolov8 algorithm,
to run the model just write this code in python:
```python
from ultralytics import YOLO
model = YOLO("omar head detection model.pt")
results = model("image.png", show=True, save=True, show_labels=True) 
```
# Examples
![image](https://github.com/omarAlharbi1/my_head_detection_yolov8_model/assets/127057011/d1b55425-fc0e-4613-aac1-006f9ed93edf)

https://github.com/omarAlharbi1/my_head_detection_yolov8_model/assets/127057011/0ad3bc05-4200-4d43-a9d5-8334605fe54b

https://github.com/omarAlharbi1/my_head_detection_yolov8_model/assets/127057011/5c18a868-8529-4771-9b9b-ddf0706a0bb0

