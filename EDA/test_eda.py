import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

# 테스트 데이터셋 로딩
coco_json_test = json.load(open('../../dataset/test.json'))
images_test = coco_json_test['images']

def random_vis_test(root_dir: str, rows: int, cols: int):
    idxs = random.sample(list(range(len(images_test))), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    for i, idx in enumerate(idxs):
        r, c = i // cols, i % cols
        img_path = os.path.join(root_dir, images_test[idx]['file_name'])
        img = Image.open(img_path).convert("RGB")

        axes[r][c].imshow(img)
        axes[r][c].axis('off')
        axes[r][c].set_title(img_path.split('/')[-1], fontsize=20)

    plt.tight_layout()
    plt.show()


# 랜덤하게 테스트 데이터 추출하기
random_vis_test("../../dataset", 2, 2)
