import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import json
import os


def run():
    st.title('Data visualization')

    with open('../dataset/train.json', 'r') as f:
        data = json.load(f)

    data_info = data['info']
    data_licenses = data['licenses']
    data_images = data['images']
    data_annotations = data['annotations']
    data_categories = data['categories']
    classes = [c['name'] for c in data_categories]


    img_path, img_id = st.selectbox(
        'choose image',
        [(img['file_name'], img['id']) for img in data_images],
    )

    st.write(f'img_path: {img_path} img_id: {img_id}')

    img = cv2.imread(os.path.join('../dataset/', img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 
    
    # get bbox
    bboxes = []
    for anno in data_annotations:
        if anno['image_id'] == img_id:
            bboxes.append(anno['bbox'])
    
    # draw bbox
    RED_COLOR = (255, 0, 0)
    LINE_WEIGHT = 2
    for bbox in bboxes:
        x, y, width, height = map(int, bbox)
        img = cv2.rectangle(img, (x, y), (x+width, y+height), RED_COLOR, LINE_WEIGHT)

    st.image(img)

    # 실행 명령어 streamlit run data_vz.py  --server.fileWatcherType none --server.port 30004

run()
# if __name__ == '__main__':
#     run()
