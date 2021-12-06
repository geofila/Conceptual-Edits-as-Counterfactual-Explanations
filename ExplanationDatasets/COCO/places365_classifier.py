import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from skimage import io


class Classifier:

    def __init__(self, arch):
        # th architecture to use
        self.arch = arch

        # load the pre-trained weights
        self.model_file = '%s_places365.pth.tar' % self.arch
        if not os.access(self.model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + self.model_file
            os.system('wget ' + weight_url)

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()


        # load the image transformer
        self.centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        
        self.classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)


    # load the test image
    def classify(self, img_url):
      img = io.imread(img_url)
      img = Image.fromarray(img)
      input_img = V(self.centre_crop(img).unsqueeze(0))

      # forward pass
      logit = self.model.forward(input_img)
      h_x = F.softmax(logit, 1).data.squeeze()
      probs, idx = h_x.sort(0, True)
      # output the prediction
      preds = []
      for i in range(0, 5):
          preds.append([self.classes[idx[i]], float(probs[i])])
      return preds
