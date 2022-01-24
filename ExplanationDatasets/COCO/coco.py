import json 
from .places365_classifier import Classifier
from tqdm.notebook import tqdm
from Queries import *

class COCO:

  def __init__(self, json_filename):
    with open (json_filename, "r") as f:
        self.json_data = json.load(f)
        self.coco = {}
    for im in self.json_data["images"]:
      self.coco[im["id"]] = {"url": im["flickr_url"]} # αρχικά αποθηκευούμε για κάθε εικόνα το url της ώστε να μπορέσουμε να το εκτυπώσουμε
    
    self.categories = {}
    for cat in self.json_data["categories"]:
      self.categories[cat["id"]] = {"name": cat["name"], "supercategory": cat["supercategory"]} 

    anns_per_image = {}
    print ("Loading Coco Metadata")
    for ann in tqdm(self.json_data["annotations"]):
      if ann["image_id"] not in anns_per_image:
        anns_per_image[ann["image_id"]] = {"label_ids": [ann["category_id"]], "label_texts": [self.categories[ann["category_id"]]["name"]]}
      else:
        anns_per_image[ann["image_id"]]["label_ids"].append(ann["category_id"])
        anns_per_image[ann["image_id"]]["label_texts"].append(self.categories[ann["category_id"]]["name"])

    # ενώνουμε τις κατηγορίες με το dict των εικόνων 
    for im_id in list(self.coco):
      if im_id in anns_per_image:
        self.coco[im_id]["label_ids"] = anns_per_image[im_id]["label_ids"]
        self.coco[im_id]["label_texts"] = anns_per_image[im_id]["label_texts"]
      else:
        self.coco.pop(im_id, None) # αν δεν έχουμε annotations σβήνουμε το key αυτό 

  def create_dataset(self, list_of_objects, predictions = None, clean_dataset = True, arch = "resnet18"):
    valid_ids = []
    if predictions == None:
        model = Classifier(arch)
    
    print ("Loading spesified subset of COCO")
    for objs in tqdm(list_of_objects):
      for im_id in self.coco:
        if set(objs).issubset(self.coco[im_id]["label_texts"]): # an eoxume classification gia thn eikona auth
          
          if predictions != None and im_id in predictions:
            self.coco[im_id]["pred_class"] = predictions[im_id]
            valid_ids.append(im_id)
            
          elif predictions == None:
            try:
                pred = model.classify(self.coco[im_id]["url"])[0]
                self.coco[im_id]["pred_class"] = pred
                valid_ids.append(im_id)
            except:
                pass
        
    if clean_dataset: 
      for im_id in list(self.coco.keys()):
        if im_id not in valid_ids:
          self.coco.pop(im_id, None)


def create_msq(coco, image_id, materialize = False):
    im = coco.coco[image_id]

    if materialize: 
        label_ids = coco.coco[image_id]["label_ids"]
        parent_labels = [frozenset([coco.categories[l]["name"], coco.categories[l]["supercategory"]]) for l in label_ids]
        concepts = parent_labels # πρέπει να προσθέοσυμε και τα facts της γνώσης 
    else:
        concepts = [frozenset([t]) for t in im["label_texts"]]
    return Query(np.array(concepts)) 
