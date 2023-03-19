import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
import os
from PIL import Image

# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
# model_file = '%s_places365.pth.tar' % arch 
model_file = 'resnet50_places365.pth.tar'

model = models.__dict__[arch](num_classes=365)
# checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
device = torch.device('cpu')
checkpoint = torch.load(model_file, map_location=device)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img_name = 'test_data/17.jpg'

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

# print('{} prediction on {}'.format(arch,img_name))
print('Prediction on {}'.format(img_name))
print(len(classes))

# output the prediction
for i in range(0, 5):
    print('{} : {:.3f}'.format(classes[idx[i]], probs[i]))
