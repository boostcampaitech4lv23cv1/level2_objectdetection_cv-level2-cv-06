import streamlit as st
import cv2
import json
import os
from typing import List


def make_checkbox(class_list: List[tuple]) -> List[int]:
    check_boxes = st.columns(5)
    return_list = [False]*len(class_list)
    for idx, (class_name, class_id) in enumerate(class_list):
        with check_boxes[idx%5]:
            check = st.checkbox(class_name, value=True)
        return_list[class_id] = check
    return return_list


def run():
    st.title('Data visualization')

    with open('../dataset/train.json', 'r') as f:
        data = json.load(f)

    data_info = data['info']
    data_licenses = data['licenses']
    data_images = data['images']
    data_annotations = data['annotations']
    data_categories = data['categories']
    classes = [(c['name'], c['id']) for c in data_categories]

    img_path, img_id = st.selectbox(
        'choose image',
        [(img['file_name'], img['id']) for img in data_images],
    )

    check_list = make_checkbox(classes)

    st.write(f'img_path: {img_path} img_id: {img_id}')

    img = cv2.imread(os.path.join('../dataset/', img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # get bbox
    bboxes = []
    for anno in data_annotations:
        if anno['image_id'] == img_id:
            bboxes.append((anno['bbox'], anno['category_id']))
    
    # draw bbox
    RED_COLOR = (255, 0, 0)
    LINE_WEIGHT = 2
    for bbox, c_id in bboxes:
        if check_list[c_id]:
            x, y, width, height = map(int, bbox)
            img = cv2.rectangle(img, (x, y), (x+width, y+height), RED_COLOR, LINE_WEIGHT)

    st.image(img)

    # 실행 명령어 streamlit run data_vz.py  --server.fileWatcherType none --server.port 30004

run()
# if __name__ == '__main__':
#     run()
