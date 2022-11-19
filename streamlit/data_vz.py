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
from collections import Counter

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

    check_list = make_checkbox(id_list=group.get_group(img_path).class_id.unique())

    st.write(f"img_path: {img_path}")

    img = cv2.imread(os.path.join("../dataset/", img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = get_bbox(group.get_group(img_path))
    img = draw_bbox(img, bboxes, check_list)

    col1, col2 = st.columns([1, 3])
    if "state" not in st.session_state:
        st.session_state.state = False

    with col1:
        if not st.session_state.state:
            st.button("choose item", on_click=change_state)
        else:
            st.button("close", on_click=change_state)

        if st.session_state.state:
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

    if st.session_state.state:
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


def make_bbox_count_tab(df: pd.DataFrame):

    """
    이미지 별 bbox 갯수 시각화
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    st.header("bbox_count")

    bbox_nums_dict = dict(df["image_id"].value_counts())
    bbox_nums = list(bbox_nums_dict.values())
    bbox_min = min(bbox_nums)
    bbox_max = max(bbox_nums)
    bbox_nums_0to9 = []
    bbox_nums_10to19 = []
    bbox_nums_20tomax = []
    for i in bbox_nums:
        if 0 <= i < 10:
            bbox_nums_0to9.append(i)
        elif 10 <= i < 20:
            bbox_nums_10to19.append(i)
        else:
            bbox_nums_20tomax.append(i)

    bbox_max_img = sorted(bbox_nums_dict.items(), key=lambda x: x[1])[-1][0]
    most_common = Counter(bbox_nums).most_common()[0]
    bbox_mode = most_common[0]
    bbox_mode_img_num = most_common[1]

    fig = plt.figure()
    plt.hist(bbox_nums, bins=bbox_max, range=(0, bbox_max))
    plt.xlabel("bbox_num")
    plt.ylabel("frequency")
    st.pyplot(fig)

    st.write(f"min_bbox: {bbox_min}")
    st.write(f"max_bbox: {bbox_max}")
    st.write(f"max_bbox: {bbox_max_img}")
    st.write(f"mode_bbox: {bbox_mode}")
    st.write(f"mode_bbox_frequency: {bbox_mode_img_num}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    ax0.hist(bbox_nums_0to9, range=(0, 10))
    ax0.set_title("bbox_nums_0to9_distribution")
    ax0.set_xlabel("bbox_num")
    ax0.set_ylabel("frequency")
    ax0.set_xticks(range(0, 10))

    ax1.hist(bbox_nums_10to19, range=(10, 20))
    ax1.set_title("bbox_nums_10to19_distribution")
    ax1.set_xlabel("bbox_num")
    ax1.set_ylabel("frequency")
    ax1.set_xticks(range(10, 20))

    ax2.hist(bbox_nums_20tomax, bins=52, range=(21, bbox_max))
    ax2.set_title("bbox_nums_over20_distribution")
    ax2.set_xlabel("bbox_num")
    ax2.set_ylabel("frequency")
    plt.tight_layout(h_pad=10)
    st.pyplot(fig)


# 실행 명령어 streamlit run data_vz.py  --server.fileWatcherType none --server.port 30004
st.set_page_config(layout="wide")
st.title("Data Visualization")
vz_tab, count_tab, bbox_count_tab = st.tabs(["analysis", "count", "bbox_count"])
df = set_data()
with vz_tab:
    label_fix_tab(df)
with count_tab:
    make_category_count_tab(df)
with bbox_count_tab:
    make_bbox_count_tab(df)

# if __name__ == '__main__':
#     run()
