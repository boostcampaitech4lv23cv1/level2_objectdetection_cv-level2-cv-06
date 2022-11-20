from pycocotools.coco import COCO
import pandas as pd
import os
import cv2
import streamlit as st
import json
import numpy as np
from typing import List
from utils.transform import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import sys
from stqdm import stqdm
from collections import Counter

CLASSES = [
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]
RED_COLOR = (255, 0, 0)
BLUE_COLOR = (0, 0, 255)
LABEL_COLORS = [
    px.colors.label_rgb(px.colors.convert_to_RGB_255(x))
    for x in sns.color_palette("Spectral", 10)
]
LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8] + LABEL_COLORS[9:]

LINE_WEIGHT = 2
TRAIN_JSON = "../dataset/train_repair.json"
CHANGED_LABELS = "../streamlit/repair.json"
STATE = False
assert os.path.exists(TRAIN_JSON), "check if train_repair.json exists"


def set_data() -> pd.DataFrame:
    """데이터 설정

    Return:
        coco format의 train.json의 annotations들을 하나의 행으로 하는 데이터프레임
    """
    coco = COCO(TRAIN_JSON)

    df = pd.DataFrame()

    image_ids = []
    class_name = []
    class_id = []
    bbox_id = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []

    for image_id in coco.getImgIds():

        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info["id"])
        anns = coco.loadAnns(ann_ids)

        file_name = image_info["file_name"]

        for ann in anns:
            image_ids.append(file_name)
            class_name.append(CLASSES[ann["category_id"]])
            class_id.append(ann["category_id"])
            bbox_id.append(ann["id"])
            x_min.append(float(ann["bbox"][0]))
            y_min.append(float(ann["bbox"][1]))
            x_max.append(float(ann["bbox"][0]) + float(ann["bbox"][2]))
            y_max.append(float(ann["bbox"][1]) + float(ann["bbox"][3]))

    df["image_id"] = image_ids
    df["class_name"] = class_name
    df["class_id"] = class_id
    df["bbox_id"] = bbox_id
    df["x_min"] = x_min
    df["y_min"] = y_min
    df["x_max"] = x_max
    df["y_max"] = y_max

    return df


def get_class_id(class_name: str) -> int:
    return CLASSES.index(class_name)


def get_class_name(class_id: int) -> str:
    return CLASSES[class_id]


def make_checkbox(id_list: List[int]) -> List[bool]:
    """
    Args:
        id_list: class_id list

    Return:
        각 클래스별 checkbox 체크 여부 list
    """
    check_boxes = st.columns(5)
    return_list = [False] * len(CLASSES)
    for idx, class_name in enumerate(CLASSES):
        with check_boxes[idx % 5]:
            class_id = get_class_id(class_name)
            if class_id in id_list:
                check = st.checkbox(class_name, value=True)
            else:
                check = st.checkbox(class_name, value=False, disabled=True)

        return_list[class_id] = check
    return return_list


def draw_aug_img(img):
    """augmentation 된 이미지를 그려주는 함수

    Args:
        img (np.array): 원본 이미지

    Returns:
        np.array: augmented 된 이미지
    """
    transformed = soft_aug()(image=img)
    return transformed["image"]


def make_aug_result_tab(df: pd.DataFrame, check_list: List[bool]):
    """사진 한 장씩 선택해서 작성한 augmentation 결과를 볼 수 있는 탭
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    st.header("Augmentation Result")

    group = df.groupby("image_id")
    img_paths = group.groups.keys()

    img_path = st.selectbox("choose images", img_paths)

    st.write(f"img_path: {img_path}")

    img = cv2.imread(os.path.join("../dataset/", img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = get_bbox(group.get_group(img_path))
    img_bbox = draw_bbox(img.copy(), bboxes, check_list)

    col1, col2 = st.columns([1, 1])

    if "open_tool" not in st.session_state:
        st.session_state.open_tool = False

    with col1:
        st.write("Original Image")
        st.image(img_bbox)

    with col2:
        aug_img = draw_aug_img(img)
        st.write("Augmented Image")
        st.image(aug_img)


def get_bbox(img_group: pd.DataFrame) -> List[list]:
    """
    Args:
        img_group: 이미지 별로 group화 된 데이터 프레임

    Returns:
        annotation id, class id 및 bbox크기를 원소로 하는 list
    """
    bboxes = []
    for _, row in img_group.iterrows():
        b_id, c_id, x_min, y_min, x_max, y_max = (
            row.bbox_id,
            row.class_id,
            row.x_min,
            row.y_min,
            row.x_max,
            row.y_max,
        )
        bboxes.append([b_id, c_id, x_min, y_min, x_max, y_max])
    return bboxes


def draw_bbox(img: np.array, bboxes: List[list], check_list: List[bool]) -> np.array:
    """
    Args:
        img: 이미지
        bboxes: bbox의 id 및 크기를 담은 list
        check_list: 클래스 가시 여부

    Return:
        check_list에 해당하는 bbox와 label들이 그려진 이미지
    """
    for bbox in bboxes:
        _, c_id, x_min, y_min, x_max, y_max = map(int, bbox)
        if check_list[c_id]:
            img = cv2.rectangle(
                img, (x_min, y_min), (x_max, y_max), RED_COLOR, LINE_WEIGHT
            )
            img = cv2.putText(
                img,
                get_class_name(c_id),
                (x_min, y_min - 10),
                cv2.FONT_ITALIC,
                1,
                RED_COLOR,
                LINE_WEIGHT,
            )
    return img


def change_label(item_id: int, pre_label: str):
    """들어온 label을 선택된 label로 바꿔주는 함수
    Args:
        item_id: 바꿀 bbox label의 id
        pre_label: 바꾸기 전의 label
    """
    label_to_change = st.selectbox("choose label to changing", CLASSES)
    current, change = st.columns(2)

    with current:
        st.write(f"current label: **{pre_label}**")
    with change:
        st.write(f"label will change: **{label_to_change}**")

    if st.button("Are you sure you want to change the label?", on_click=show_log):

        class_id = get_class_id(label_to_change)

        with open(TRAIN_JSON, "r") as f:
            data = json.load(f)

        with open(TRAIN_JSON, "w") as f:
            data["annotations"][item_id]["category_id"] = class_id
            json.dump(data, f, indent=2)

        if not os.path.exists(CHANGED_LABELS):
            with open(CHANGED_LABELS, "w") as f:
                json.dump({}, f, indent=2)

        with open(CHANGED_LABELS, "r") as f:
            log_data = json.load(f)
        with open(CHANGED_LABELS, "w") as f:
            log_data.update({str(item_id): class_id})
            json.dump(log_data, f, indent=2)


def change_state():
    try:
        st.session_state.state = not st.session_state.state
    except Exception:
        print("not exist session variable")


def show_log():
    st.write("modify complete!")


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
    sns.histplot(bbox_nums)
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

    ax0.set_title("bbox_nums_0to9_distribution")
    ax0.set_xlabel("bbox_num")
    ax0.set_xticks(range(10))
    ax1.set_title("bbox_nums_10to19_distribution")
    ax1.set_xlabel("bbox_num")
    ax1.set_xticks(range(10, 20))
    ax2.set_title("bbox_nums_over20_distribution")
    ax2.set_xlabel("bbox_num")

    sns.histplot(bbox_nums_0to9, ax=ax0)
    sns.histplot(bbox_nums_10to19, ax=ax1)
    sns.histplot(bbox_nums_20tomax, ax=ax2)
    plt.tight_layout(h_pad=10)
    st.pyplot(fig)


def make_bboxes_proportion_tab():
    """
    이미지 별 이미지의 bboxes가 이미지 전체에서 차지하는 비율 분포 시각화
    """

    img = plt.imread("bboxes_proportion.PNG")
    image = np.array(img)
    st.image(image)


def display_plotly_figs(figs_path: str):
    """plotly figures를 streamlit page에 표시하는 함수, plotly figure list가 저장된 pickle 파일을 불러옴
    Args:
        color_dist_figs_path: pickle 파일 경로
    """
    try:
        with open(figs_path, "rb") as fr:
            color_dist_figs = pickle.load(fr)
    except Exception:
        sys.stderr.write("No file: %s\n" % figs_path)
        exit(1)
    for color_dist_fig in color_dist_figs:
        st.plotly_chart(color_dist_fig)


def make_color_dist_tab(df: pd.DataFrame, figs_path: str):
    """bbox 내의 color distribution의 box plot 표시
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
        color_dist_figs_path: color distribution figure pickle 파일 경로
    """
    if st.button("refresh"):
        make_color_dist_figs(df, figs_path)
    display_plotly_figs(figs_path)


def make_color_dist_figs(df: pd.DataFrame, figs_path: str):
    """bbox 내의 color distribution의 box plot 계산 및 저장
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    color_list = ["r_mean", "g_mean", "b_mean", "h_mean", "s_mean", "v_mean"]
    group = df.groupby("image_id")
    img_paths = list(group.groups.keys())
    len_df = len(df)
    color_ann_cumulation = {color: [0] * len_df for color in color_list}

    for img_path in stqdm(img_paths):
        img_bgr = cv2.imread(os.path.join("../dataset/", img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bboxes = get_bbox(group.get_group(img_path))
        for bbox in bboxes:
            b_id, c_id, x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_rgb = img_rgb[y_min:y_max, x_min:x_max]
            rgb_mean = np.mean(cropped_rgb, axis=(0, 1))
            cropped_hsv = img_hsv[y_min:y_max, x_min:x_max]
            hsv_mean = np.mean(cropped_hsv, axis=(0, 1))
            color_mean = np.concatenate((rgb_mean, hsv_mean))
            for i, color in enumerate(color_list):
                color_ann_cumulation[color][b_id] = color_mean[i]

    for color in color_list:
        df[color] = color_ann_cumulation[color]

    fig_list = []
    for color in color_list:
        fig = px.box(
            df.sort_values(by="class_name"),
            x="class_name",
            y=color,
            color="class_name",
            color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING,
            notched=True,
            labels={"class_name": "Class Name", "frac_bbox_area": "BBox Area (%)"},
            title="<b>Class 별 이미지 내 Bbox 의 " + color + " 분포 </b>",
        )
        fig.update_layout(
            showlegend=True,
            yaxis_range=[-10, 260],
            legend_title_text=None,
            xaxis_title="",
            yaxis_title="<b> " + color + " </b>",
        )
        fig_list.append(fig)

        with open(figs_path, "wb") as fw:
            pickle.dump(fig_list, fw)


def make_bboxes_size_prop_tab(df: pd.DataFrame):

    df["frac_bbox_area"] = (
        (df["x_max"] - df["x_min"]) * (df["y_max"] - df["y_min"]) / 1024 / 1024 * 100
    )

    fig = px.box(
        df.sort_values(by="class_name"),
        x="class_name",
        y="frac_bbox_area",
        color="class_name",
        color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING,
        notched=True,
        labels={"class_name": "Class Name", "frac_bbox_area": "BBox Area (%)"},
        title="<b>Class 별 이미지 내 Bbox 크기 비율 분포</b>",
    )

    fig.update_layout(
        showlegend=True,
        yaxis_range=[-0.25, 101.5],
        legend_title_text=None,
        xaxis_title="",
        yaxis_title="<b>Bbox 크기 비율 (%)</b>",
    )
    st.plotly_chart(fig)
