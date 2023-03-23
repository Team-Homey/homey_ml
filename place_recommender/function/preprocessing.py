from torch.autograd import Variable as V
from torchvision import transforms
from PIL import Image
import os
import io

def transform_image(img_bytes) : 
    transform_img = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return transform_img(img).unsqueeze(0)