# Rule
> Rule : BlackFormatter로 설정
> - 클래스명 : PascalCase
> - 함수명, 변수명 : snake_case
> - 각 함수, 클래스 사이 두 줄 띄우기
> - annotation(type hint) 사용
> - 간단한 docstring 작성
> - 기능 만들 때 feat/기능 내용으로 브랜치 명 작성
> - 데이터 분석시 EDA/분석 내용으로 브랜치 명 작성
> 
> ex)
> ```py
> "feat/gethyperparams"
> "EDA/rgb_analysis"
> ```

> Commit Convention
> - feat, fix, docs, refactor, test, remove, add
> - 세부 내용 적기
> (fix는 오류or버그 고쳤을 때, refactor는 코드 수정했을 때)
> 
> ex)
>  ```bash
>  "feat: get model hyper parameter"
>  "세부 내용 추가"
>  ```

---
# 프로젝트 report
바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cae4a605-5922-4459-a474-0a1e9e55843b/Untitled.png)

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

- **Input :** 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 됩니다. bbox annotation은 COCO format으로 제공됩니다.
- **Output :** 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다.
- 프로젝트 팀 구성 및 역할

| 팀원 / 역할 | streamlit | MMDetection | YOLO | Paper Review |
| --- | --- | --- | --- | --- |
| 오주헌 | Augmentation,EDA Tab | VFNet, DETR, ATSS | YOLOv7, YOLOv5 | YOLOv7 |
| 강민수 |  | Swinv2
Cascade-Rcnn
ATSS
Parameter Tuning
WBF | WBF | model soup  |
| 신성윤 | Color Distribution EDA Tab | Faster R-CNN
Cascade R-CNN
VFNet
EfficientDet | YOLOv5 | Cascade R-CNN |
| 나성근 | EDA 탭 제작 /  | Centernet
label smooth loss | YOLOv5 학습 | WBF |
| 박시형 | Dataset 수정 툴, Class 분포 Tab | DETR, VFnet, Cornernet | Yolov7 | FPN |

![EX) Streamlit Data Label 수정](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/008fec0d-357f-4a70-85b9-107365892385/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.48.09.png)

EX) Streamlit Data Label 수정

![EX) Streamlit EDA (Bbox_size_prop)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/52a79f8c-2375-453b-bbec-e687235dfada/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-12-02_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.45.47.png)

EX) Streamlit EDA (Bbox_size_prop)

![초기 대회 전략 브레인 스토밍](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ed0b10a6-8169-4e35-a66d-29838758f741/Untitled.png)

초기 대회 전략 브레인 스토밍

![최종 대회 전략](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/84d30816-e899-42db-a359-d1d613b2088c/Untitled.png)

최종 대회 전략

### 프로젝트 수행 절차 및 방법

1. streamlit을 이용해서 EDA 진행
    1. EDA 과정을 팀원들과 쉽게 공유
    2. Data Label 수정 작업 진행
2. MMDetection을 이용해서 2stage-baseline으로 정함
    1. 여러 모델 실험 (DETR, Cascade R-CNN, VFNet, ATSS)
3. Backbone을 Swinv2으로 정함 (base 기준으로 swinv2가 swin에 비해 0.015 정도 성능 향상)
4. YOLOv7을 1stage-baseline으로 정함
5. 앙상블 전략 실험(WBF)

- EDA
    - 데이터 직접 살펴보기
        - 통계 분석 과정에서 확인하기 어려운 데이터의 특성 파악
        - class 분류 기준 구체화
        - label 수정(적용 x)
    - Statistical Analysis
        - Class distribution
        - Bbox number distribution per image
        - Bbox sizes distribution
        - Bbox sizes distribution per image by class
        - Color distribution by class
- Model Architecture
    - 2 stages
        - Detector: Faster R-CNN, Cascade R-CNN, VFNet, DETR, ATSS
        - Backbone: ResNet, Swinv2
    - 1 stage
        - Detector: YOLOv5, YOLOv6, YOLOv7
        - Backbone: v5 (s,x), v6 (s), v7 (e6e)
- Augmentation

|  | Geometric (p=1) | Color (p=0.5) | Noise (p=0.4) | Histogram Equalize (p=0.5) |
| --- | --- | --- | --- | --- |
| 2stage
Augmentation | Multi-scailing Training(512~1024 / by 32)
RandomRotate90 (p=1.0)
Random Flip (p=0.5) | • Random Brightness Contrast (p=0.5)
• RGB shift (p=0.5) | • Blur (p=0.5)
• Gaussian Noise (p=0.5)
• Image Compression (p=0.5) | • CLAHE  (p=0.5)
• Saturation (p=0.5)
• Random Gamma (p=0.5) |
| 1stage
Augmentation | configs |  |  |  |
- Hyperparameter Tuning

|  | Confidence Score Threshold | NMS | Optimizer | LR Scheduler | TTA | Input size |
| --- | --- | --- | --- | --- | --- | --- |
| 1stage | 0.001 | 0.5 | SGD | Cosine | Resize: 1,0.83,0.67
Flip: Horizontal, Vertical | 1280 |
| 2stage | 0.00 | 0.5 | Adamw
(lr=1e-4) | CosineAnnealing
(min_lr=5e-6) + warmup(3 epoch) | Resize: 512x512, 640x640, 768x768, 896x896, 1024x1024, randomflip | 1024 |
- Ensemble(Final submission)
    - Weighted Box Fusion
        
        5-fold Cascade R-CNN with TTA + 5-fold YOLOv7 with TTA + 5-fold YOLOv7 without TTA
        

### 프로젝트 수행 결과

- 핵심 실험 내용

| 실험 내용 | 결과 |
| --- | --- |
| baseline(Faster-RCNN) | 0.4195 |
| swin_base + cascade_rcnn | 0.55 |
| swinv2_base + cascade_rcnn | 0.57 |
| swinv2_base+cascade_rcnn+augmentation | 0.59 |
| swinv2_base+cscade_rcnn+augmentation+img_size(1024) | 0.61 |
| swinv2_large+cascade+rcnn+augmentation+img_size(multi_scale/512~1024)+TTA(flip,multiscale) | 0.643 |
| 5-Fold Ensemble(WBF) | 0.677 |
| YOLOv7_e6e | 0.638 |
| 5-Fold YOLOv7_e6e(TTA) + 5-Fold swinv2 Cascade (WBF) | 0.7064 |
| 5-Fold YOLOv7_e6e(TTA)+5-Fold YOLOv7_e6e(wo TTA)+5-Fold swinv2 Cascade | 0.7075 |
- 최종 제출: Public:0.7075(3rd) / Private:0.6970(3rd)

### 자체 평가 의견

- 학습 시간이 오래 걸려서 다양한 시도를 하지 못했다.
- 실험 process의 자동화를 더 고려하지 못했다.
- 학습 시간을 줄이는 시도를 하지 못했다. (INPUT사이즈를 다양하게 주는 실험)
- 전체적인 파이프라인을 잘 알지 못해 계획을 제대로 세우지 못했다.
- MMDetection이 작동되는 구조를 파악하는데 시간이 오래 걸렸다.
- Model의 parameter수, FLOPS 값을 파악하고 대회에 활용했으면 더 좋았을 것이다.
- 데이터의 일관성에 관한 문제를 해결하지 못하였다.
- Bounding Box에 초점을 맞춰서 개선을 했지만, classification 부분도 더 고려했으면 좋았을 것이다.
