import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_gdn import GDN
from os import environ
# environ['CUDA_VISIBLE_DEVICES']='0'
device = t.device('cuda')
# gdn = GDN(n_ch, device)
# igdn = GDN(n_ch, device,inverse = True)
class GaNet(nn.Module):
    def __init__(self):
        super(GaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5,stride = 2, padding = 2)
        self.conv2 = nn.Conv2d(128, 128, 5,stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(128, 192, 5, stride=2, padding=2)
        self.gdn128 = GDN(128,device)

    def forward(self, x):
        x = self.gdn128(self.conv1(x))
        x = self.gdn128(self.conv2(x))
        x = self.gdn128(self.conv2(x))
        x = self.conv3(x)
        return x

class GsNet(nn.Module):
    def __init__(self):
        # the last channel:3*16*16  /96    output 3*256*256
        super(GsNet, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.Conv2d(192, 192 * 2, 5, stride=1, padding=2),
            GDN(192 * 2, device, inverse=True),
            nn.Conv2d(192 * 2, 192 * 4, 5, stride=1, padding=2),
            GDN(192 * 4, device, inverse=True),
            nn.PixelShuffle(16)
        )

    def forward(self, img):
        return self.conv_transpose(img)