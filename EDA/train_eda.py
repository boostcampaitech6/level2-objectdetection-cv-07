import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# train.json 파일을 읽어 데이터 프레임으로 만들기 
root = "../../dataset/"

with open(root + "train.json") as f:
    train = json.load(f)

categories = pd.DataFrame(train['categories'])
ann_df = pd.DataFrame(train['annotations'])
img_df = pd.DataFrame(train['images'])

# annotation 데이터 프레임에 image_id를 기준으로 file_name을 추가하기
ann_df['file_name'] = ann_df['image_id'].map(img_df['file_name'])
train.keys()

# images 정보
train['images'][0].keys()
# annotations 정보
train['annotations'][0].keys()

print('{}개의 image'.format(len(train['images'])))
print('{}개의 category'.format(len(train['categories'])))
print('{}개의 annotation'.format(len(train['annotations'])))

class_name = categories['name'].unique()
print(f'쓰레기 종류: {class_name}')
ann_df.head(10)

# 각 이미지별로 몇개의 bbox가 있는지 확인
ann_df['image_id'].value_counts().describe()
# 각 카테고리별로 몇개의 bbox가 있는지 확인
ann_df['category_id'].value_counts()

# 각 카테고리별로 몇개의 bbox가 있는지 시각화
sns.set(rc={'figure.figsize':(15,10)})
ctgr_bbox_df = pd.concat([ann_df['image_id'], ann_df['category_id']], axis=1)

ctgr_bbox_df['Categories'] = ctgr_bbox_df['category_id'].map(categories['name'])

bat_plot = sns.countplot(y='Categories', 
                         data=ctgr_bbox_df, 
                         order=ctgr_bbox_df['Categories'].value_counts().index)
bat_plot.set_title('Number of Bounding Box per Category', fontsize=20)

for i, p in enumerate(bat_plot.patches):
    bat_plot.annotate(ctgr_bbox_df['Categories'].value_counts()[i], 
                      (p.get_x()+p.get_width()+50, p.get_y()+0.6), 
                      ha='center', 
                      va='center', 
                      xytext=(10, 10), 
                      textcoords='offset points')

# 제일 큰 bbox와 제일 작은 bbox, 그리고 평균 bbox의 크기를 확인
print(f'max area: {ann_df.area.max()} \
      \nmin area: {ann_df.area.min()} \
      \nmean area: {ann_df.area.mean()}\n')

# 제일 큰 bbox와 제일 작은 bbox를 가진 이미지의 정보
ann_df[ann_df['area'].isin([ann_df.area.max(), ann_df.area.min()])]

# bbox를 그리기 위한 색 리스트
# General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
colors = ['yellow', 'blue', 'green', 'red', 'orange', 'magenta', 'pink', 'cyan', 'gray', 'olive']

# bbox를 그리는 메소드
def draw_bbox(img_id, lt = 500, gt = 1000000, all = False,):
    '''
    img_id: 이미지 id
    lt: 크기가 lt이하인 bbox 그리기
    gt: 크기가 gt이상인 bbox 그리기
    all: True일 경우 모든 bbox를 그리고 False일 경우 lt와 gt에 해당하는 bbox만 그리기
    '''

    img = Image.open(root + ann_df[ann_df['image_id'] == img_id]['file_name'].values[0])
    draw_df = ann_df[ann_df['image_id'] == img_id].drop(['image_id', 'id', 'iscrowd', 'file_name'], axis=1)

    print(f"\n{img_id}번 이미지 category별로 bbox개수 확인\n", draw_df.groupby('category_id').count())

    # 제일 작은 박스가 위로 오도록 정렬
    draw_df.sort_values(by='bbox', inplace=True)    

    for i, (category, (x_min, y_min, w, h)) in enumerate(zip(draw_df['category_id'], draw_df['bbox'])):
        if all: # 모든 bbox를 그리기
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle([(x_min, y_min), 
                            (x_min+w, y_min+h)], 
                            outline=colors[category], 
                            width=3,
                            fill=(0, 0, 0, 50))
        elif draw_df['area'].values[i] <= lt or draw_df['area'].values[i] >= gt: # bbox의 크기가 lt보다 작거나 gt보다 큰 경우만 그리기
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle([(x_min, y_min), 
                            (x_min+w, y_min+h)], 
                            outline=colors[category], 
                            width=3,
                            fill=(0, 0, 0, 50))
    img.show(img)

# 제일 작은 bbox를 가진 이미지와 제일 큰 bbox를 가진 이미지를 bbox를 함께 시각화
for idx in [1063, 1160]:
    draw_bbox(idx, 500, 1000000, True)

# bbox의 크기가 500보다 작은 이미지들의 정보
ann_df[ann_df['area'] < 500].sort_values(by='area')

# area가 3.12인 bbox를 가진 이미지 시각화
draw_bbox(1377, 500, 1000000)

# area가 66.25인 bbox를 가진 이미지 시각화
draw_bbox(3712, 500, 1000000)

# bbox의 크기가 1000,000보다 큰 이미지들의 정보
ann_df[ann_df['area'] > 1000000].sort_values(by='area', ascending=False)

# area가 1,000,000보다 큰 bbox를 가진 이미지 시각화
draw_bbox(802, 500, 100000, True)

# Category별 bbox의 평균 크기를 확인
avg_area = pd.concat([pd.DataFrame({'Categories': class_name}), 
                      ann_df.groupby('category_id')['area'].mean()], axis=1)

avg_area.sort_values(by='area', ascending=False, inplace=True)
avg_area

# 각 카테고리별로 bbox의 평균 넓이를 시각화
fig, ax = plt.subplots(figsize=(15,10))

barplot = sns.barplot(x='Categories', y='area', data=avg_area)
barplot.set_title('Average Area per Category', fontsize=20)

for i, p in enumerate(barplot.patches):
    barplot.annotate(int(avg_area['area'][i]), 
                      (p.get_x() + 0.3, p.get_y()+p.get_height()), 
                      ha='center', 
                      va='center', 
                      xytext=(10, 10), 
                      textcoords='offset points')
