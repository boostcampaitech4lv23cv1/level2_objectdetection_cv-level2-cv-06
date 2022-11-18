import streamlit as st
import cv2
import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import sys

sys.path.append("../")
from utils.streamlit_utils import *


def label_fix_tab(df: pd.DataFrame):
    """사진 한 장씩 선택해서 원하는 카테고리의 bbox를 선택해서 볼 수 있는 탭
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    st.header("Data Analysis")

    group = df.groupby("image_id")
    img_paths = group.groups.keys()

    img_path = st.selectbox("choose image", img_paths)

    check_list = make_checkbox(id_list=df.class_id.unique().tolist())

    st.write(f"img_path: {img_path}")

    img = cv2.imread(os.path.join("../dataset/", img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = get_bbox(group.get_group(img_path))
    img = draw_bbox(img, bboxes, check_list)

    col1, col2 = st.columns([1, 3])

    with col1:
        state = st.button("choose item")
        if state:
            idx, selected_id, selected_item = st.radio(
                "Choose data what you change",
                [(idx, b[0], CLASSES[b[1]]) for idx, b in enumerate(bboxes)],
                format_func=lambda x: f"bbox_id: {x[1]} | class_name: {x[2]}",
            )
            _, _, x_min, y_min, x_max, y_max = map(int, bboxes[idx])
            img = cv2.rectangle(
                img, (x_min, y_min), (x_max, y_max), BLUE_COLOR, LINE_WEIGHT
            )
            img = cv2.putText(
                img,
                selected_item,
                (x_min, y_min - 10),
                cv2.FONT_ITALIC,
                1,
                BLUE_COLOR,
                LINE_WEIGHT,
            )

    with col2:
        st.image(img)

    if state:
        change_label(selected_id, selected_item)


def make_category_count_tab(df: pd.DataFrame):
    """카테고리 별 bbox 갯수 시각화
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    st.header("category_count")
    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x=df.class_name)
    st.pyplot(fig)


# 실행 명령어 streamlit run data_vz.py  --server.fileWatcherType none --server.port 30004
st.set_page_config(layout="wide")
st.title("Data Visualization")
vz_tab, count_tab = st.tabs(["analysis", "count"])
df = set_data()
with vz_tab:
    label_fix_tab(df)
with count_tab:
    make_category_count_tab(df)
# if __name__ == '__main__':
#     run()
