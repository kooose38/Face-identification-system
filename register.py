from PIL import Image 
import pickle 
import argparse 

from src.utils.load_net import load_net 
from src.utils.detect import detect 

def register(img_path: str):
    '''
    画像からベクトル変換し、リストに登録する関数
    前提としてfacenet_pytorch.MTCNNでは複数人を同時に検知しないので、単独の顔写真を読み込む
    '''
    regist_path = "./db/"
    net, mtcnn, _  = load_net()
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img).unsqueeze(0) # 自動的にfacenetの入力形式に変換される (b, c, w, h)
    out = detect(face, net)
    # ベクトルをデータとして登録する
    with open(regist_path+"register.pkl", "wb") as f:
        pickle.dump([out], f)
    print("登録が完了しました。")
    
parser = argparse.ArgumentParser(description='自分の顔を登録する画像ファイルを入力してください')
parser.add_argument('--image', help='image file path', type=str, default="./img/sample/sample_register.jpg")
args = parser.parse_args()
register(str(args.image))