import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pseudo labeling하고 싶은 csv파일 불러오기
df = pd.read_csv("../../baseline/faster_rcnn/faster_rcnn_torchvision_submission.csv")
print(df.head(8))

import pandas as pd

# 새로운 리스트 생성
data_list = []

# 띄어쓰기로 나눠진 데이터를 6개씩 끊어서 리스트에 추가
for index, row in df.iterrows():
    image_id = row['image_id']
    prediction_string = str(row['PredictionString'])
    split_data = prediction_string.split()
    num_chunks = len(split_data) // 6
    for i in range(num_chunks):
        data_list.append({
            'image_id': image_id,
            'label': split_data[i * 6],
            'score': split_data[i * 6 + 1],
            'xmin': split_data[i * 6 + 2],
            'ymin': split_data[i * 6 + 3],
            'xmax': split_data[i * 6 + 4],
            'ymax': split_data[i * 6 + 5],
        })

# 리스트를 DataFrame으로 변환
new_df = pd.DataFrame(data_list)

# 결과 확인
print(new_df.head())
print(new_df.tail(5))

label_mapping = {
    '0': 'General trash',
    '1': 'Paper',
    '2': 'Paper pack',
    '3': 'Metal',
    '4': 'Glass',
    '5': 'Plastic',
    '6': 'Styrofoam',
    '7': 'Plastic bag',
    '8': 'Battery',
    '9': 'Clothing'
}

# 라벨 번호를 이름으로 매핑
new_df['label_name'] = new_df['label'].map(label_mapping)

new_df.to_csv('your_output_file.csv', index=False)

print(new_df.tail())

# 라벨(label)별 이미지의 갯수 계산
label_image_counts = new_df.groupby('label_name')['image_id'].nunique().reset_index(name='image_count')
label_image_counts_sorted = label_image_counts.sort_values(by='image_count', ascending=False)

print(label_image_counts_sorted)

# 바 그래프 그리기
ax =sns.barplot(x='label_name', y='image_count', data=label_image_counts_sorted, order=label_image_counts_sorted['label_name'])
plt.title('Number of Images per Label')
plt.xlabel('Label')
plt.ylabel('Number of Images')

for index, value in enumerate(label_image_counts_sorted['image_count']):
    ax.text(index, value, str(value), ha='center', va='bottom')


# x 축 레이블 설정
plt.xticks(rotation=45, ha='right', fontsize=10)  # x 축 레이블 회전 및 정렬, 폰트 크기 조절
plt.tight_layout()  # 레이아웃 조정

plt.show()

# 라벨(label)별 Bounding Box의 갯수 계산
label_bbox_counts = new_df.groupby('label_name').size().reset_index(name='bbox_count')
# 'bbox_count' 열을 기준으로 내림차순 정렬
label_bbox_counts_sorted = label_bbox_counts.sort_values(by='bbox_count', ascending=False)
print(label_bbox_counts_sorted)

# 바 그래프 그리기
ax=sns.barplot(x='label_name', y='bbox_count', data=label_bbox_counts_sorted, order=label_bbox_counts_sorted['label_name'])
plt.title('Number of Bounding Boxes per Label')
plt.xlabel('Label')
plt.ylabel('Number of Bounding Boxes')

for index, value in enumerate(label_bbox_counts_sorted['bbox_count']):
    ax.text(index, value, str(value), ha='center', va='bottom')

# x 축 레이블 설정
plt.xticks(rotation=45, ha='right', fontsize=10)  # x 축 레이블 회전 및 정렬, 폰트 크기 조절
plt.tight_layout()  # 레이아웃 조정

plt.show()


import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random


def visualize_bbox(image_path, bboxes, labels):
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # bounding box 시각화
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 이미지 출력
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# CSV 파일에서 데이터 로드
csv_path = '../../level2-objectdetection-cv-07/EDA/your_output_file.csv'
df = pd.read_csv(csv_path)

# 이미지의 파일 경로
random_image = random.choice(df['image_id'].unique())
random_image_path = '../../dataset/'+random_image

# 선택한 이미지의 데이터 필터링
selected_data = df[df['image_id'] == random_image]

# 선택한 이미지의 bounding box와 라벨
selected_bboxes = selected_data[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)
selected_labels = selected_data['label_name'].tolist()

# 출력을 위한 디버깅 정보
print("random_image:", random_image_path)
print("Number of Bounding Boxes:", len(selected_bboxes))
print("Unique Labels:", set(selected_labels))

# 시각화 함수 호출
visualize_bbox(random_image_path, selected_bboxes, selected_labels)
