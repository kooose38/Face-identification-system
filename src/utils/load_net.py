import torch 
from facenet_pytorch import InceptionResnetV1, MTCNN
from mtcnn.mtcnn import MTCNN as MT

from src.networks.facenet import AsiaFaceNet

def load_net():
    '''
    モデルの読み込み
    '''
    net = AsiaFaceNet()
    net.load_state_dict(torch.load("./weights/facenet1.pth", map_location={"cuda:0": "cpu"}))
    net.eval()
    mtcnn = MT() # 複数人を同時検知
    mtcnn_for_facenet = MTCNN() # 単独で検知
    return net, mtcnn_for_facenet, mtcnn # No GPU