import streamlit as st
import cv2
import numpy as np
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml') 

st.title("ตรวจจับวัตถุ : ภาพนิ่ง")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
    pred_image, obj_box = yolo.predictions(img)
    
    if len(obj_box) > 0:
        obj_names = ''
        for obj in obj_box:
            obj_names = obj_names + obj[4] + ' '
        text_obj = 'ตรวจพบ ' + obj_names
    else:
        text_obj = 'ไม่พบวัตถุ'
    #----------------------------------------------
    st.header(text_obj)
    st.image(pred_image, caption='ภาพ Output',channels="BGR")
    
