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

    nums = []
    max_bbox = 0
    max_img = 0
    for image_id in df["image_id"].unique():
        num = len(df[df["image_id"] == image_id])
        nums.append(num)
        if num > max_bbox:
            max_img = image_id
            max_bbox = num

    dic = dict()
    for num in nums:
        if num not in dic.keys():
            dic[num] = 1
        else:
            dic[num] += 1
    mode = sorted(dic.items(), key=lambda x: x[1])[-1][0]

    st.write(f"min_bbox_num: {min(nums)}")
    st.write(f"max_bbox_num: {max(nums)}")
    st.write(f"mode_bbox_num: {mode}")
    st.write(f"max_bbox_img: {max_img}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    ax0.hist(nums, bins=70)
    ax0.set_title("total_distribution")
    ax0.set_xlabel("bbox_num")
    ax0.set_ylabel("image_num")

    ax1.hist(nums, bins=20, range=(0, 21))
    ax1.set_title("1~20_distribution")
    ax1.set_xlabel("bbox_num")
    ax1.set_ylabel("image_num")

    ax2.hist(nums, bins=50, range=(21, max(nums)))
    ax2.set_title("over21_distribution")
    ax2.set_xlabel("bbox_num")
    ax2.set_ylabel("image_num")

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
