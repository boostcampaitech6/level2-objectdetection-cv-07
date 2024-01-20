import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np
import json

# Built In Imports
from datetime import datetime
from glob import glob
import warnings
import IPython
import urllib
import zipfile
import pickle
import shutil
import string
import math
import tqdm
import time
import os
import gc
import re

# Visualization Imports
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
import matplotlib
import plotly
import PIL
import cv2
from pycocotools.coco import COCO

# Other Imports
from tqdm.notebook import tqdm

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
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    # 제일 작은 박스가 위로 오도록 정렬
    draw_df.sort_values(by='bbox', inplace=True)    
    ax = plt.gca()
    for i, (category, (x_min, y_min, w, h)) in enumerate(zip(draw_df['category_id'], draw_df['bbox'])):
        if all: # 모든 bbox를 그리기
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle([(x_min, y_min), 
                            (x_min+w, y_min+h)], 
                            outline=colors[category], 
                            width=3,
                            fill=(0, 0, 0, 50))
            ax.text(x_min, y_min - 10 , class_name[category], weight = 'bold', color = 'tomato')
        elif draw_df['area'].values[i] <= lt or draw_df['area'].values[i] >= gt: # bbox의 크기가 lt보다 작거나 gt보다 큰 경우만 그리기
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle([(x_min, y_min), 
                            (x_min+w, y_min+h)], 
                            outline=colors[category], 
                            width=3,
                            fill=(0, 0, 0, 50))
            ax.text(x_min, y_min - 10 , class_name[category], weight = 'bold', color = 'tomato')

    plt.imshow(img)

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

LABEL_COLORS = [px.colors.label_rgb(px.colors.convert_to_RGB_255(x)) for x in sns.color_palette("Spectral", 10)]
LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8]+LABEL_COLORS[9:]

coco = COCO(root + "train.json")

train_df = pd.DataFrame()

image_ids = []
class_name = []
class_id = []
x_min = []
y_min = []
x_max = []
y_max = []
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
for image_id in coco.getImgIds():
        
    image_info = coco.loadImgs(image_id)[0]
    ann_ids = coco.getAnnIds(imgIds=image_info['id'])
    anns = coco.loadAnns(ann_ids)
        
    file_name = image_info['file_name']
        
    for ann in anns:
        image_ids.append(file_name)
        class_name.append(classes[ann['category_id']])
        class_id.append(ann['category_id'])
        x_min.append(float(ann['bbox'][0]))
        y_min.append(float(ann['bbox'][1]))
        x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
        y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

train_df['image_id'] = image_ids
train_df['class_name'] = class_name
train_df['class_id'] = class_id
train_df['x_min'] = x_min
train_df['y_min'] = y_min
train_df['x_max'] = x_max
train_df['y_max'] = y_max

# 한 이미지에 존재하는 유니크한 카테고리 개수 시각화
fig = px.histogram(train_df.groupby('image_id')["class_id"].unique().apply(lambda x: len(x)), 
             log_y=True, color_discrete_sequence=['skyblue'], opacity=0.7,
             labels={"value":"Number of Unique class"},
             title="<b>DISTRIBUTION OF # OF Unique Class PER IMAGE   " \
                   "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
                   )
fig.update_layout(showlegend=False,
                  xaxis_title="<b>Number of Unique CLASS</b>",
                  yaxis_title="<b>Count of Unique IMAGE</b>",)
fig.show()

# 이미지에서 bbox가 주로 위치하는 부분을 시각화
bbox_df = pd.DataFrame()
bbox_df['class_id'] = train_df['class_id'].values
bbox_df['class_name'] = train_df['class_name'].values
bbox_df['x_min'] = train_df['x_min'].values / 1024
bbox_df['x_max'] = train_df['x_max'].values / 1024
bbox_df['y_min'] = train_df['y_min'].values / 1024
bbox_df['y_max'] = train_df['y_max'].values / 1024
bbox_df['frac_x_min'] = train_df['x_min'].values / 1024
bbox_df['frac_x_max'] = train_df['x_max'].values / 1024
bbox_df['frac_y_min'] = train_df['y_min'].values / 1024
bbox_df['frac_y_max'] = train_df['y_max'].values / 1024

ave_src_img_height = 1024
ave_src_img_width = 1024

# DEFAULT
HEATMAP_SIZE = (ave_src_img_height, ave_src_img_width, 14)

# Initialize
heatmap = np.zeros((HEATMAP_SIZE), dtype=np.int16)
bbox_np = bbox_df[["class_id", "frac_x_min", "frac_x_max", "frac_y_min", "frac_y_max"]].to_numpy()
bbox_np[:, 1:3] *= ave_src_img_width
bbox_np[:, 3:5] *= ave_src_img_height
bbox_np = np.floor(bbox_np).astype(np.int16)

# Color map stuff
custom_cmaps = [
    matplotlib.colors.LinearSegmentedColormap.from_list(
        colors=[(0.,0.,0.), c, (0.95,0.95,0.95)], 
        name=f"custom_{i}") for i,c in enumerate(sns.color_palette("Spectral", 10))
]
custom_cmaps.pop(8) # Remove No-Finding

for row in tqdm(bbox_np, total=bbox_np.shape[0]):
    heatmap[row[3]:row[4]+1, row[1]:row[2]+1, row[0]] += 1
    
fig = plt.figure(figsize=(20,25))
plt.suptitle("Heatmaps Showing Bounding Box Placement\\n ", fontweight="bold", fontsize=16)
for i in range(10):
    plt.subplot(4, 4, i+1)
    if i==0:
        plt.imshow(heatmap.mean(axis=-1), cmap="bone")
        plt.title(f"Average of All Classes", fontweight="bold")
    else:
        plt.imshow(heatmap[:, :, i-1], cmap=custom_cmaps[i-1])
        plt.title(f"{classes[i-1]} – ({i})", fontweight="bold")
        
    plt.axis(False)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# 카테고리 별 bbox의 aspect ratio 값

# Aspect Ratio is Calculated as Width/Height
bbox_df["aspect_ratio"] = (bbox_df["x_max"]-bbox_df["x_min"])/(bbox_df["y_max"]-bbox_df["y_min"])

# Display average means for each class_id so we can examine the newly created Aspect Ratio Column
display(bbox_df.groupby("class_id").mean())

# Generate the bar plot
fig = px.bar(x=classes, y=bbox_df.groupby("class_id").mean()["aspect_ratio"], 
             color=classes, opacity=0.85,
             color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING, 
             labels={"x":"Class Name", "y":"Aspect Ratio (W/H)"},
             title="<b>Aspect Ratios For Bounding Boxes By Class</b>",)
fig.update_layout(
                  yaxis_title="<b>Aspect Ratio (W/H)</b>",
                  xaxis_title=None,
                  legend_title_text=None)
fig.add_hline(y=1, line_width=2, line_dash="dot", 
              annotation_font_size=10, 
              annotation_text="<b>SQUARE ASPECT RATIO</b>", 
              annotation_position="bottom left", 
              annotation_font_color="black")
fig.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="red", opacity=0.125,
              annotation_text="<b>>2:1 VERTICAL RECTANGLE REGION</b>", 
              annotation_position="bottom right", 
              annotation_font_size=10,
              annotation_font_color="red")
fig.add_hrect(y0=2, y1=3.5, line_width=0, fillcolor="green", opacity=0.04,
              annotation_text="<b>>2:1 HORIZONTAL RECTANGLE REGION</b>", 
              annotation_position="top right", 
              annotation_font_size=10,
              annotation_font_color="green")
fig.show()