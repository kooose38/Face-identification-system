import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np 
import os, time, uuid
from glob import glob 
from tqdm import tqdm 
import cv2 
from PIL import Image 
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean 
from typing import Dict, Any, Union 
from src.utils.load_net import load_net 
from src.utils.transform import transform
import argparse

from src.utils.detect import detect

def discrimination(img_path: str, threshold=0.40) -> Dict[str, float]:
    '''
    複数人が移っている画像から登録されている画像と同人物を検知して写真として返す
    '''
    with open("./db/register.pkl", "rb") as f:
        register_v = pickle.load(f) # (1, 512)
    # 複数人を同時に検知したいのでMTCNNを用いる
    # こちらは自動でテンソルに変換しないので自身の前処理クラスで形状を整える
    net, _, mtcnn = load_net()
    # 描画オブジェクトの生成
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    id = uuid.uuid4()

    # cas_file = "./Github/face/kasumi/haarcascade_frontalface_alt.xml"
    # cas = cv2.CascadeClassifier(cas_file)
    sample_img = cv2.imread(img_path)
    imgs = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    # face_list = cas.detectMultiScale(img_gray)
    face_list = mtcnn.detect_faces(imgs)
    results = {"id": "detected_"+str(id)[:2]}
    for i, faces in enumerate(face_list): # 検知された顔画像の分だけループする
        # 座標の取得
        x = faces["box"][0]
        y = faces["box"][1]
        w = faces["box"][2]
        h = faces["box"][3]
        # 顔領域の切り出し
        face = cv2.cvtColor(sample_img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB) # (w, h, c)
        # 前処理
        face = Image.fromarray(face)
        face_tensor = transform("val", face).unsqueeze(0)
        # 推論によるvector化
        out = detect(face_tensor, net).reshape(1, -1)
        # 類似度
        sim = cosine_similarity(register_v, out)[0][0]
        color = (0, 255, 0) if sim >= threshold else (0, 0, 255)
        # 判定を基に矩形で囲む
        cv2.rectangle(sample_img, (x, y), (x+w, y+h), color, thickness=2)
        cv2.putText(sample_img, text=str(i), org=(x, y), color=color, thickness=1, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0)
        # アノテーションの作成
        result = {
            "face": [x, y, w, h],
            "discrimination": 1 if sim >= threshold else 0, 
            "similar": sim
        }
        results[f"{str(i)}"] = result
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig(f"./result/detected_{str(id)[:2]}.png")
    print(f"./result/detected_{str(id)[:2]}に保存しました")
    return results 


parser = argparse.ArgumentParser(description='認証したい画像を入力してください')
parser.add_argument('--image', help='image file path', type=str, default="./img/sample/sample_recognized.jpg")
parser.add_argument('--threshold', help='recognized thresholds', type=float, default=0.40)
args = parser.parse_args()
result = discrimination(str(args.image), float(args.threshold))
print(result)
