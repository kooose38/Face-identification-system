import torch.nn as nn 
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN

# 参照書籍: PytorchではじめるAI開発 6章から一部抜粋
class AsiaFaceNet(nn.Module):
    def __init__(self):
        super(AsiaFaceNet, self).__init__()
        '''
        モデルそのままでは人種による精度低下がみられるのでアジア人種のデータセットで転移学習をする
        なお、損失関数として年齢と性別を設ける。
        '''
        self.base = InceptionResnetV1(pretrained="vggface2")
        # 最終層のみ学習させる
        for name, weight in self.base.named_parameters():
            if name in ["last_linear.weight", "last_bn.weight", "last_bn.bias"]:
                weight.requires_grad = True 
            else:
                weight.requires_grad = False 
        self.age = nn.Linear(512, 1)
        self.sex = nn.Linear(512, 2)

    def forward(self, x):
        y = self.base(x)
        age = F.hardtanh(self.age(y), 15, 75)
        sex = self.sex(y)
        return y, age, sex 