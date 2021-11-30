from torch import nn

class RenderingNet(nn.Module):
    def __init__(self, channels_in=22, channels_out=3):
        super(RenderingNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels_in, 16, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels_out, kernel_size=(3, 3), padding=1, bias=True)
        )

    def forward(self, tensor_in):
        return self.net(tensor_in)
