from torchvision import transforms

class Transform():
    def __init__(self, resize=160):
        '''推論時の顔を検証した画像それぞれに対して前処理する関数'''
        self.data_trans = {
            "val": transforms.Compose([
                                     transforms.Resize((resize, resize)),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x*2-1)  
            ])
        }

    def __call__(self, phase, img):
        return self.data_trans[phase](img) # (3, 160, 160)
    
transform = Transform()