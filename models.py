import torch
import torch.nn as nn

class DINOv3(nn.Module):
    def __init__(self, MODEL, DIR, CKPT):
        super(DINOv3, self).__init__()
        self.model = torch.hub.load(DIR, MODEL, source='local', weights=CKPT)
        self.last_layer = 12

    def forward(self, x):
        pass

    def inference(self, x):
        y = self.model.get_intermediate_layers(x, n=self.last_layer, reshape=True, norm=True)[-1]
        return y


if __name__ == '__main__':
    model = DINOv3(MODEL='dinov3_vits16', DIR='/torch/dinov3', CKPT='/torch/ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth').to("cuda")
    x = torch.randn((1,3,256,256), dtype=torch.float32).to("cuda")
    y = model.inference(x)
    print(y.shape)
