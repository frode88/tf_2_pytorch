import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling'
]


class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature learning/Convolution Layer
        self.conv = nn.Sequential(
            #Convolution + relu
            # conv2d input, output, kernel , stripe and padding

            # Stride of the convolving kernel
            # padding is used to control the amount of padding applied to the input
            nn.Conv2d(3, 64, kernel_size =(3, 3), stride = 1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(64, 64, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            # Pooling - Max Pooling
            # Max pooling is a pooling operation that selects the maximum\
            # element from the region of the feature map covered by the filter
            # This will help avoid overfiting
            # kernal size 2 and stride 2
            nn.MaxPool2d(2, 2, stride=2),

            #Convolution + relu
            nn.Conv2d(64, 128, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(128, 128, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            # Pooling
            nn.MaxPool2d(2, 2, stride=2),

            #Convolution + relu
            nn.Conv2d(128, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 256, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(256, 512, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(512, 512, kernel_size =(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            #Convolution + relu
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),

        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            # Linear equl to Dense layer
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            # in_features – size of each input sample
            # out_features – size of each output sample
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 604),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    model = PHOSCnet()

    x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250)

    y = model(x)

    print(y['phos'].shape)
    print(y['phoc'].shape)
