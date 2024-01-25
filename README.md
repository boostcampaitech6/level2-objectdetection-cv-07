# [Pstage] CV 07조 BIG-I 👁️
<p align="center">
  <img width="1191" alt="스크린샷 2024-01-25 오후 9 45 25" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-07/assets/83398511/16ee58b2-0015-48c8-9a55-65c9eb388348">
</p>

- 프로젝트명: 재활용 품목 분류를 위한 Object Detection
- 프로젝트 전체 기간 (3주): 2024년 1월 3일 (수요일) 10:00 ~ 2024년 1월 18일 (목요일) 19:00
---
# 목표

일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기를 Detection하는 모델을 개발

---

# 전체 데이터셋 통계

- 전체 이미지 개수 : 9754장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)

---

# 학습 데이터

학습데이터는 train.json과 train 디렉토리로 제공됨

- train: 4883장의 train image 존재
- train.json: train image에 대한 annotation file (coco format)
    - [coco format](https://cocodataset.org/#home)은 크게 2가지 (`images`, `annotations`)의 정보를 가지고 있습니다.

---
    
# 평가데이터
    
본 대회의 결과물 csv 확장자 파일을 제출하게 됨
    
- input: 쓰레기를 촬영한 이미지 4871장
- output: 각 이미지에 대한 class, bounding box 좌표값

---

# 결과값
    
각 이미지에서 총 10개의 클래스를 예측하고 이미지에 포함된 객체 별 score(confidence)와
bounding box(X_min, X_max, Y_min, Y_max) 그리고 이미지 이름을 출력해야함
    
예시:
    
| PredictionString | image_id |
| --- | --- |
| 0 0.59 432.18 352.39 644.98 668.7 1 0.44 436.45 347.73 648.38 679.3  | test/0020.jpg |

---

# 평가 방식

- Test set의 mAP50(Mean Average Precision)로 평가
    - Object Detection에서 사용하는 대표적인 성능 측정 방법
    - Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단합니다.
    - metric
      <p align="center">
        <img width="650" alt="precision" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-07/assets/83398511/e3f1024b-d51f-4140-8e41-34056fe12edd">
        <img width="650" alt="recall" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-07/assets/83398511/0b010f26-f6f5-4691-8bab-3d1e502d75f8">
        <img width="650" alt="mAP" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-07/assets/83398511/17a763eb-2a56-4c48-b5bc-fd331b709587">
      </p>

---

# 프로젝트 팀 구성 및 역할

  | 김한규 | 민하은 | 이하연 | 심유승 | 안채연 | 강동기 | 
  | :-: | :-: | :-: | :-: | :-: | :-: |
  | <img width="100" src="https://avatars.githubusercontent.com/u/32727723?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/87661039?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/83398511?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/86558738?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/146207162?v=4"> | <img width="100" src="https://avatars.githubusercontent.com/u/121837927?v=4"> |

- **김한규**: 모델 선정, Straitified Kfold, confusion matrix, ensemble
- **민하은**: 모델 선정 - 2 stage detector (Cascade rcnn), Hyper Parameters Tuning, ensemble
- **이하연**: Train EDA, 문제 분석, 모델 선정, Hyper Parameters Tuning, Straitified Kfold, confusion matrix, ensemble
- **심유승**: Train EDA, 모델 선정 - 1 stage detector (RetinaNet), Hyper Parameters Tunning
- **안채연**: Test EDA, baseline code 가이드, 모델 선정, Hyper Parameters Tuning, emsemble
- **강동기**:  모델 선정, Data preprocessing, Data oversampling, Hyper parameter Tuning, Kfold, softvoting, confusion matrix, ensemble

---

# 수행 절차 및 방법
<p align="center">
  <img width="491" alt="schedule" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-07/assets/83398511/3e3ba6f9-97bf-4b2f-9db8-1d03e0402c49">      
</p>      

  1. 프로젝트 개발환경 구축 (GitHub, Slack, Server, Notion)      
  2. 문제 분석과 베이스라인 코드 이해와 협업을 위한 가이드 작성      
  3. EDA를 진행하며 주어진 문제와 데이터의 구조를 분석하고 이해      
  4. 성능 향상을 위한 데이터 전처리 방식 탐색      
  5. One-stage 및 Two-Stage 계열 모델들을 나누어 담당하여 코드 작성 및 성능 분석      
  6. Augmentation, loss 및 lr 조정을 통한 Hyper Parameters Tuning       
  7. oversampling및  Straitified Kfold을 통한 성능 향상        
  8. confusion matrix를 기반한 앙상블을 진행하여 최종 결과물 제작

---

# 모델 실험

1. wbf 앙상블 모델 5개와 단일 모델 1개를 wbf로 앙상블
    1. 앙상블 모델
        1. Deformable-DETR, Faster R-CNN, YOLOv8, YOLO-NAS l avg, YOLO-NAS l best(스티로폼 over sampling 추가), YOLO-NAS m  6 모델 wbf
        2. Cascade R-CNN(wbf), EfficientDet, Faster R-CNN, RetinaNet, YOLO-NAS l, YOLOv6, YOLOv8 7 모델 wbf
        3. Faster R-CNN, YOLOv8, EfficientDet, YOLOv6, YOLO-NAS l avg, YOLO-NAS l best 6 모델 wbf
        4. YOLOv8(이미지 사이즈 512) 5-fold wbf
        5. YOLOv8(이미지 사이즈 1024) 4-fold wbf
    2. 단일 모델: YOLOv8 4-fold 이미지 사이즈 1024
2. wbf 앙상블 모델 5개를 wbf로 앙상블
    1. 앙상블 모델은 위 1. a. 앙상블 모델과 동일

---
# 수행 결과
| | public → private |
| --- | --- |
| mAP | 0.6523 → 0.6395 |
| mAP | 0.6522 → 0.6393 |
