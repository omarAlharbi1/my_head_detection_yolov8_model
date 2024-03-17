from ultralytics import YOLO
import os
import shutil


model = YOLO("best.pt")

images_location="auto_annotation_tool/images/"
labels_location="auto_annotation_tool/labels/"

def copy_image_to(source_file,destination):
    # directory_path = os.path.dirname(source_file)
    if not os.path.exists(destination):
        os.makedirs(destination)

    shutil.copy(source_file, destination)

def save_boxes(location,file_name,results):
    if not os.path.exists(location):
        os.makedirs(location)

    file_name,_=os.path.splitext(file_name)
    file_name = str(location)+file_name+".txt"

    for box in results[0].boxes.cpu().numpy():
        box_vector=box.xywhn[0]
        print(box_vector)

        with open(file_name, "a") as file:
            text_box_vector = "0 "+ str(box_vector[0])+" "+ str(box_vector[1])+" "+ str(box_vector[2])+" "+ str(box_vector[3])
            file.write(text_box_vector + "\n")
        print("================================================================")

my_data_folder="auto_annotation_tool/my_raw_data/"
for image in os.listdir(my_data_folder):
    print(image)
    results = model(my_data_folder+image)
    save_boxes(labels_location,image,results)
    copy_image_to(my_data_folder+image,images_location)
    print("=================================================================================================")
    print("=================================================================================================")


