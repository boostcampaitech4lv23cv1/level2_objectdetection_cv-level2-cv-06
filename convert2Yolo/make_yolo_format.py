import shutil
import os
from glob import glob

# step 0: example 실행

os.makedirs("../dataset/yolo_train/train", exist_ok=True)
os.makedirs("../dataset/yolo_val/train", exist_ok=True)
os.makedirs("../dataset/yolo_test/test", exist_ok=True)


os.system(
    'python example.py --dataset COCO --img_path ../dataset --label ../stratify_dataset/train_fold_0.json --convert_output_path ../dataset/yolo_train --img_type ".jpg" --manifest_path ../dataset/yolo_train --cls_list_file ./name.txt'
)

os.system(
    'python example.py --dataset COCO --img_path ../dataset --label ../stratify_dataset/val_fold_0.json --convert_output_path ../dataset/yolo_val --img_type ".jpg" --manifest_path ../dataset/yolo_val --cls_list_file ./name.txt'
)

# os.system(
#     'python example.py --dataset COCO --img_path ../dataset --label ../dataset/test.json --convert_output_path ../dataset/yolo_test --img_type ".jpg" --manifest_path ../dataset/yolo_test --cls_list_file ./name.txt'
# )

# subprocess.run(
#     [
#         "python",
#         "example.py",
#         "--dataset",
#         "COCO",
#         "--img_path",
#         "../dataset",
#         "--label",
#         "../stratify_dataset/train_fold_0.json",
#         "--convert_output_path",
#         "../datatset/yolo_train",
#         "--img_type",
#         '".jpg"',
#         "--cls_list_file",
#         "./name.txt",
#     ]
# )

# step 1: yolo_(state)에 포함된 이미지만 복사해서 옮기기


def get_index(path: str) -> str:
    return path.split("/")[-1][:-4]


def get_yolo_path(path: str, state: str) -> str:
    return os.path.join(f"../dataset/yolo_{state}/", path.split("/")[-1])


# train, val
img_paths = glob("../dataset/train/*.jpg")
train_fold_paths = glob("../dataset/yolo_train/train/*.txt")
val_fold_paths = glob("../dataset/yolo_val/train/*.txt")

train_fold_lst = [get_index(p) for p in train_fold_paths]
val_fold_lst = [get_index(p) for p in val_fold_paths]

for img_path in img_paths:
    if get_index(img_path) in train_fold_lst:
        shutil.copyfile(img_path, get_yolo_path(img_path, "train"))

    if get_index(img_path) in val_fold_lst:
        shutil.copyfile(img_path, get_yolo_path(img_path, "val"))


# test
test_img_paths = glob("../dataset/test/*.jpg")

test_lst = [get_index(p) for p in test_img_paths]

for test_path in test_img_paths:
    shutil.copyfile(test_path, get_yolo_path(test_path, "test"))


# step 2: yolo_train/train txt파일 밖으로 빼기
train_txt_paths = glob("../dataset/yolo_train/train/*.txt")
val_txt_paths = glob("../dataset/yolo_val/train/*.txt")

for txt_path in train_txt_paths:
    shutil.move(txt_path, get_yolo_path(txt_path, "train"))

for txt_path in val_txt_paths:
    shutil.move(txt_path, get_yolo_path(txt_path, "val"))
