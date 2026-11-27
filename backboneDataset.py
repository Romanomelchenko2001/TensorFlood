from torchvision.transforms import v2
from torchvision import tv_tensors
import transformers
transformers.logging.set_verbosity_info()
transformers.utils.logging.enable_progress_bar()

from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import kagglehub
import datasets
import torch
from torchvision import transforms
import math
from datasets import load_dataset
from torcheval.metrics.functional import multiclass_f1_score

from transformers import Trainer
# Download latest version


from transformers import default_data_collator

def merge_ships(objs):
    #print(objs)
    #print(objs.name)
    return pd.DataFrame(
        {
         "ImageId":objs.name,
         "EncodedPixels":[list(objs.EncodedPixels.values)] if objs.EncodedPixels.values.shape[0]>1 else objs.EncodedPixels.values
         }
    )

class BackBoneDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, datasetPaths, requiresMaskedLabels,
                turn_prob = 0.5,
                prob_configs = {"hor_flip":0.4,
                                 "vertical_flip":0.3,
                                 "gaussian_blur":0.5},
                image_shape = {
                    "h":768,
                    "w":768,
                    "c":3
                },
                path_obj = None
                ):
        
        
        self.datasetPaths = datasetPaths ## add random permutation
        self.requiresMaskedLabels = requiresMaskedLabels
        self.probabilities = prob_configs
        self.HWChannels = image_shape
        self.turnProb = turn_prob
        super().__init__()
        
        # create selecting positive and negative examples from the dataset
        positive_part = self.datasetPaths.EncodedPixels.index[self.datasetPaths.EncodedPixels.notna()]
        negative_part = self.datasetPaths.EncodedPixels.index[~self.datasetPaths.EncodedPixels.notna()]
        self.training_paths = np.concatenate((np.random.choice(positive_part, size = int(positive_part.shape[0]*0.95), replace=False),
        np.random.choice(negative_part, size = int(negative_part.shape[0]*0.95), replace=False)))
        np.random.shuffle(self.training_paths)

        
        self.testing_paths = np.array( list(set(np.asarray(self.datasetPaths.index)).difference(set(self.training_paths))))
        self.path_obj = path_obj

    def _info(self):
        return datasets.DatasetInfo(
            description="training dataset for cv task",
            features = datasets.Features({
            "path": datasets.Value("string"),
            "images": datasets.Array3D(shape=(self.HWChannels["c"],
                                             self.HWChannels["h"],
                                             self.HWChannels["w"]),
                                    dtype="float32"),
            "labels": datasets.Value("int8") if not self.requiresMaskedLabels else \
                                    datasets.Array3D(shape=(1,
                                             self.HWChannels["h"],
                                             self.HWChannels["w"]),
                                    dtype="float32")
            }))

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs = {
                                        "split": datasets.Split.TRAIN, 
                                        "indices": self.training_paths,
                                        "imageId_list":self.datasetPaths.loc[self.training_paths].ImageId,
                                        "EncodedPixels_list": self.datasetPaths.loc[self.training_paths].EncodedPixels
                                    }),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs = {
                                        "split": datasets.Split.TEST,
                                        "indices": self.testing_paths,
                                        "imageId_list":self.datasetPaths.loc[self.testing_paths].ImageId,
                                        "EncodedPixels_list": self.datasetPaths.loc[self.testing_paths].EncodedPixels
                                    }
                                    )
        ]
        

    def _generate_examples(self, split, indices, imageId_list, EncodedPixels_list):
        # yield and random seed
        # self.paths = random permute

        # create callback on epoch end(if streaming) -> after processing a certain amount of steps, to randomize training examples
        # worker_seed = torch.initial_seed() % 2**32
        # np.random.seed(worker_seed)
        #random.seed(worker_seed)

                ####

        # ADD RANDOMIZATION OF INDEXES ON WHICH I AM ITERATING, JUST ORDER OF Is IN THE RANGE

        ####
        def create_mask_with_gaussian(image, coords):
            
            image_ = np.zeros_like(image[:, :, 1]).astype(np.uint8)
            print(coords)
            if coords is not None:
                # coords_ = coords[coords[:, 0].argsort()]
                # coords_ = coords_[coords_[:, 1].argsort()]
                kernels = np.random.randint(5, 15)*np.ones(2).astype(int)*2+1
                if isinstance(coords, list):
                    for coord in coords:
                        image_ = cv2.fillPoly(image_, [coord], color=(255, 255, 255))
                else:
                    image_ = cv2.fillPoly(image_, [coords], color=(255, 255, 255))
                
                image_ = cv2.GaussianBlur(image_.astype(float), ksize=kernels, sigmaX=kernels[0], sigmaY=kernels[1])
            return image_

        def bounding_box(item_):
            coords_ = item_.split(" ")
            #coords_ = np.array(coords_span[::2]).astype(int)
            #coords_.sort()
            item = [(int(c)//image.shape[1], int(c)%image.shape[1]) for c in coords_[::2]]
            spans = [int(sp) for sp in coords_[1::2]]
            item = item + [(item[i][0], item[i][1]+spans[i]) for i in range(len(item))]
            
            item = np.asarray(item)
            y = item[item[:, 1].argmin()].reshape(-1, 2)
            y_min = y[y[:, 0].argmin()]

            x = item[item[:, 0].argmin()].reshape(-1, 2)
            x_min = x[x[:, 1].argmax()]

            x = item[item[:, 1].argmax()].reshape(-1, 2)
            x_max = x[x[:, 0].argmin()]

            y = item[item[:, 0].argmax()].reshape(-1, 2)
            y_max = y[y[:, 1].argmax()]
            return np.array((y_min, x_min, x_max, y_max))
        dataset_indices_shuffled = np.arange(imageId_list.shape[0])
        if split == "train":
            np.random.shuffle(dataset_indices_shuffled)
        for i in dataset_indices_shuffled:
            #print()
            label = EncodedPixels_list.iloc[i]

            image = cv2.imread(self.path_obj / "train_v2" / imageId_list.iloc[i])

            if isinstance(label, list):
                bounding_box_ = [bounding_box(label_) for label_ in label]
            else:
                bounding_box_ = bounding_box(label) if not pd.isna(label) else None

            max_dist = None
            if self.requiresMaskedLabels:
                
                target = create_mask_with_gaussian(image, bounding_box_)
                #target = target[:, :, 0]
            
            else:
                #print(pd.isna(label))
                label__ = pd.isna(label)
                target = int(not label__ if not(isinstance(label__, np.ndarray)) else not label__[0])
                #print(target)


            if isinstance(label, list):
                                
                                #target = target[:, :, 0]
                target_means = [np.mean(targ, axis=0) for targ in bounding_box_]
                center = np.asarray([[image.shape[0]//2, image.shape[1]//2]])
                max_dist = np.argmax(np.linalg.norm(
                    target_means - center, axis=-1).flatten())
   
            if split == 'train':
                if bounding_box_ is not None:
                    if max_dist is not None:
                        bounding_box_ = bounding_box_[max_dist]
                    
                    mass = np.mean(bounding_box_, axis=0)
                    #print(mass, max_dist, bounding_box_)
                    if self.turnProb >= np.random.uniform(0, 1):
                        distances = np.asarray([[image.shape[0]//2,image.shape[1]], [image.shape[0]//2, 0], [0, image.shape[1]//2], [image.shape[0], image.shape[1]//2]])
                        #min_distance = np.linalg.norm(bounding_box_[:, :, None] - np.ones((4, *bounding_box_.shape))*bounding_box_, axis=-1)
                        degr = (-5+ np.random.random()*10)/57.29
                    
                        min_ = np.argmin(np.linalg.norm(np.transpose(np.ones((4, *bounding_box_.shape))*bounding_box_, axes = (1, 0, 2)) - np.ones((4, *bounding_box_.shape))*distances, axis=-1).flatten())
                        min_point = min_//4
                        min_side = min_%4
                        v1 = bounding_box_[min_point]-mass
                        v2_ = distances[min_side]-mass
                        theta = np.arccos(v1@v2_ / (np.linalg.norm(v1)*np.linalg.norm(v2_)))
                        catet2 = v1 * np.sin(theta)
                        max_theta = np.arctan2(np.linalg.norm(catet2), np.linalg.norm(v2_))
                        turn_angle = (theta - max_theta, degr)[np.argmin([abs(theta - max_theta), abs(degr)])]
                    else:turn_angle=0
                else: 

                    turn_angle = (-5+ np.random.random()*10)/57.29 if self.turnProb >= np.random.uniform(0, 1) else 0
                    mass = (self.HWChannels["h"]//2,self.HWChannels["w"]//2)
            
                #cv2.line(, [0,0], np.array(image.shape[:2])-1, color=(255,255,255))
            
                rotation_matrix = cv2.getRotationMatrix2D(mass, turn_angle*57.29, 1)
            
                image_turned = cv2.warpAffine(image.copy(), M = rotation_matrix, dsize = image.shape[:2])
                mask_turned = target if isinstance(target, int) else cv2.warpAffine(target.copy(), M = rotation_matrix, dsize = image.shape[:2])
                
                augmentation = v2.Compose([
                    v2.ColorJitter(
                        brightness=(0.7, 1.5),
                        contrast=(0.7, 1.5),
                        saturation=(0.5, 1.7),
                        hue=(-0.3, 0.3),
                    ),
                    v2.RandomHorizontalFlip(p=0.4),#4
                    v2.RandomVerticalFlip(p=0.3),#3
                    v2.RandomApply(
                        [v2.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))],
                        p=0.5,
                    ),
                ])
                #augmentation.

                image = tv_tensors.Image(
                    torch.tensor(image_turned)
                        .permute(2, 0, 1)
                        .float() / 255
                )

                mask = mask_turned if isinstance(mask_turned, int) else tv_tensors.Mask(
                    torch.tensor(mask_turned).float() / 255
                )
                #print(isinstance(mask, tv_tensors.Mask), mask_turned)
                res = augmentation({
                    "image": image,
                    "mask": mask[None, :, :]
                }) if isinstance(mask, tv_tensors.Mask) else {"image":augmentation(image), "mask":mask}
                image, target = res["image"], res["mask"]
                # target = mask_turned if isinstance(mask_turned, int) else \
                # augmentation(torch.tensor(mask_turned).float() / 255)
            else: 
                image = torch.tensor(image).permute(2, 0, 1).float() / 255
                target = target if isinstance(target, int) else \
                    torch.tensor(target).float() / 255
            #print(imageId_list.iloc[i])
            yield indices[i], {
                "path":imageId_list.iloc[i],
                "images":image,
                "labels":target
            }


def get_dataset(sample_size=30000, segmentation_masks = False):

    path = kagglehub.competition_download('airbus-ship-detection')

    print("Path to competition files:", path)
    path_obj = Path(path)
    train_segmentations = pd.read_csv(path_obj / "train_ship_segmentations_v2.csv")
    res = train_segmentations.groupby("ImageId").apply(merge_ships).rename(columns={"ImageId":"ImageId0"}).reset_index().drop(["ImageId0", "level_1"], axis=1)
    res = res.iloc[:sample_size]
    train_segmentations = res
    obj = train_segmentations[~train_segmentations.EncodedPixels.isna()].iloc[1]
    image = cv2.imread(path_obj / "train_v2" / obj.ImageId)

    ds = BackBoneDataset(train_segmentations, segmentation_masks, path_obj=path_obj)

    train_ds = ds.as_streaming_dataset(split="train")
    test_ds = ds.as_streaming_dataset(split="test")
    return train_ds, test_ds