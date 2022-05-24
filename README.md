# CNN
# Pest Image Classification with CNN using PyTorch
The dataset discussed is image classification for pests, the name of the plant pest is hydrangea. We can get this data from kaggle.
# Import Package
import common packages:

**import numpy as np**

**import matplotlib.pyplot as plt**

import PyTorch's common packages

**import torch**

from **torch** import **nn, optim**

from **jcopdl.callback** import **Callback, set_config**

from **torchvision** import **datasets, transforms**

from **torch.utils.data** import **DataLoader**

**checking for GPU/CPU**

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader

![Screenshot 2022-05-24 205328](https://user-images.githubusercontent.com/86812576/170052617-f91965c6-96bf-440b-afdf-571db710c38f.png)

_batch_size_ will use 128 images. _crop_size_ is 64, because the input from CNN must always be the same so it must be determined at the beginning what size you want to crop. 64 is a small size, but I did this so that the running model is not too heavy, but ideally is 224.

_ImageFolder_ is the place where the data will be loaded, its structure is _"folder_name"/"file_name"_. do you think we want to shuffle the testloader? actually if it's not shuffled it's okay, but because at the end it will do a sanity check so I want the data randomly.

![torch pipeline](https://user-images.githubusercontent.com/86812576/170059694-235a78c3-ed9a-40af-b57e-03217dd88111.png)

Next, torchvision already has a default data pipeline.
data/train, after that there will be a data pipeline until finally the image size is according to our needs, which is 64x64, and this is only one image, later there will be a dataloader that automatically pulls only a total of 128 images (batch size). And because our images are colored so the total will be 128 images, 3 channels, 64 rows, and 64 columns or NCHW, this is the format on CNN, but remember not to flatten because later it will be convoluted.

So that every time you pull a minibatch of 128 images, it's a never-before-seen image because it's augmented. For example, the augmentation of rotation is 15 degrees, after that it is randomly resized by 64 cropping, then it will be randomly flipped horizontally or vertically, and finally converted to a tensor. So data augmentation just plugs in the pipeline, and when the DataLoader wants to pull data, it will process the argumentation.

Don't augment test data, because it's predictable, don't do it. only do crop_size so that the size fits the input because the number of features is already fit.
The trick is to resize to 70, then CenterCrop to 64. why do you need to resize? because the image size is different depending on the device in which the image is taken.
