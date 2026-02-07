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

class CIFARNet(nn.Module):
    def __init__(self, n_classes=100):
        super(CIFARNet, self).__init__()
        CHANNELS = [64, 128, 256, 512]

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=CHANNELS[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CHANNELS[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=CHANNELS[0], out_channels=CHANNELS[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CHANNELS[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=CHANNELS[1], out_channels=CHANNELS[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CHANNELS[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=CHANNELS[2]*4*4, out_features=CHANNELS[3]),
            nn.ReLU(),
            nn.Linear(in_features=CHANNELS[3], out_features=n_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

if __name__ == '__main__':
    model = DINOv3(MODEL='dinov3_vits16', DIR='/torch/dinov3', CKPT='/torch/ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth').to("cuda")
    x = torch.randn((1,3,256,256), dtype=torch.float32).to("cuda")
    y = model.inference(x)
    print(y.shape)

    _model = CIFARNet()
    _x = torch.randn((4,3,32,32))
    _y = _model(_x)
    print(_y.shape)
