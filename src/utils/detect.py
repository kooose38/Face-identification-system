import torch 
import numpy as np 
def detect(img: torch.Tensor, net) -> np.ndarray:
    '''
    入力は前処理済の顔画像
    shape (b, c, w, h)
    '''
    with torch.no_grad():
        out, _ , _ = net(img) # 推論時には性別、年齢の予測は使わない
        out = out[0].detach().cpu().numpy() # (512, )
    return out 