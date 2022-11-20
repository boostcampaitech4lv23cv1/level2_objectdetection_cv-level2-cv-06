import streamlit as st
import cv2
import os
import pandas as pd
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

    return check_list


# 실행 명령어 streamlit run data_vz.py  --server.fileWatcherType none --server.port 30004
st.set_page_config(layout="wide")
st.title("Data Visualization")
(
    vz_tab,
    count_tab,
    bbox_count_tab,
    bboxes_proportion_tab,
    bboxes_size_prop,
    color_tab,
    aug_tab,
) = st.tabs(
    [
        "analysis",
        "count",
        "bbox_count",
        "bbox_proportion",
        "bboxes_size_prop",
        "color_distribution",
        "augmentation",
    ]
)
df = set_data()
with vz_tab:
    check_list = label_fix_tab(df)
with count_tab:
    make_category_count_tab(df)
with bbox_count_tab:
    make_bbox_count_tab(df)
with bboxes_proportion_tab:
    make_bboxes_proportion_tab()
with bboxes_size_prop:
    make_bboxes_size_prop_tab(df)
with color_tab:
    color_dist_figs_path = "./color_dist_figs.pkl"
    make_color_dist_tab(df, color_dist_figs_path)
with aug_tab:
    make_aug_result_tab(df, check_list)

# if __name__ == '__main__':
#     run()
