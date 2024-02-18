
import torch
from torchvision import datasets
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np

class ApexDetection(datasets.VisionDataset):

    def __init__(self, root: str, split = "train", transforms= None) -> None:

        super().__init__(root, transforms)
        self.split = split
        self.root = root
        self.ids = [i for i in range(len(self.root))]
        self.transforms = transforms

    def _load_image(self, idx: int):
      image = self.root[idx]['image']
      image = tv_tensors.Image(image)
      image = image.float() / 255.0
      return image

    def _load_target(self, idx: int):
      target = self.root[idx]['objects.category']
      target = torch.tensor(target, dtype=torch.int64)
      return target
    
    def _load_box(self, idx: int, img):
      bbox = self.root[idx]['objects.bbox']
      bbox = tv_tensors.BoundingBoxes(bbox, format="XYWH", canvas_size=F.get_size(img))
      bbox = torchvision.ops.box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
      return bbox

    def __getitem__(self, idx:int):
      image = self._load_image(idx)
      bbox = self._load_box(idx, image)
      label = self._load_target(idx)

      target = {}
      target["boxes"] = bbox
      target["labels"] = label
      target["image_id"] = torch.tensor(self.root[idx]['image_id'], dtype=torch.int64)
      target["area"] = torch.tensor(self.root[idx]['objects.area'], dtype=torch.int64)
      target["iscrowd"]= torch.zeros((len(label),), dtype=torch.int64)

      if self.transforms is not None:
            image, target = self.transforms(image, target)

      return image, target

    def __len__(self):
        return len(self.ids)
    
    def __prediction__(self, img, model):
       img = img.permute(0,1,2).unsqueeze(0)
       model.eval()
       prediction = model(img)
       print(prediction)
       prediction = {k: item[:2] for k,item in prediction[0].items()}
       return prediction


    def do_plot(self, idx:int, ax=None, model=None):
      
      image, target = self. __getitem__(idx)
      boxes, labels, image_id = target['boxes'], target['labels'], target['image_id']
      scores = [None for _ in range(len(boxes))]


      if ax is None:
        ax = plt.gca()

      ax.set_axis_off()
      ax.set_title("Image_id: "+str(image_id.detach().numpy()), fontsize=8, loc='left')
      bp = ax.imshow(image.squeeze(0).permute(1, 2, 0), aspect="auto")

      colors = ['#B8911C', '#D7AA23'] if model else ['#60DF89','#3BD6C6']

      if model:
        prediction = self.__prediction__(image, model)
        boxes, labels = prediction['boxes'].detach().numpy(), prediction['labels'].detach().numpy()
        scores = prediction['scores'].detach().numpy()

      for box, label, s  in zip(boxes, labels, scores):
        x,y,w,h = box
        color = colors[0] if label==1 else colors[1]
        
        rect = patches.Rectangle((x, y), w-x, h-y, linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        text = np.round(s, 2) if model else 'id:'+str(label.detach().numpy())
        ax.text(x, y, f'{str(text)}', ha='left', va='bottom', color=color, fontsize=7, fontstyle='italic', fontweight="bold")


      return bp
      