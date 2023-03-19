import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

# from torch.autograd import Variable as V
# from torchvision import transforms
# from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.arch = 'resnet50'
        self.model_file = 'models/%s_places365.pth.tar' % self.arch
        self.classes_file = 'models/categories_places365.txt'
        
        self.activation = dict()
        self.classes = list()
        
        self.model = models.__dict__[self.arch](num_classes=365)
        device = torch.device('cpu')
        checkpoint = torch.load(self.model_file, map_location=device)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.model.avgpool.register_forward_hook(self.get_activation('avgpool'))
        
        with open(self.classes_file) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)
        
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):
        with torch.no_grad():
            output = self.model(x)
        return self.activation['avgpool'].reshape(-1,2048)
    
    def prediction(self, x):
        with torch.no_grad():
            x = x.view(x.size(0), -1)
            fc = nn.Linear(512 * 4, 365)
            x = self.model.fc(x)
            h_x = F.softmax(x, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
        return self.classes[idx[0]], probs[0]