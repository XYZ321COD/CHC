import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, level_number=4, args=None):
        super(Model, self).__init__()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.tree_model = nn.Sequential(nn.Linear(2048, ((2**(level_number+1))-1) - 2**level_number), nn.Sigmoid())
        
        if args.load_model:
            self.load_state_dict(torch.load('./pre-trained/128_0.5_200_128_1000_model.pth', map_location='cpu'), strict=False)
            print("Load the model")
        self.masks_for_level = {level: torch.ones(2**level).cuda() for level in range(1, 4+1)}


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        tree_output = self.tree_model(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), tree_output
