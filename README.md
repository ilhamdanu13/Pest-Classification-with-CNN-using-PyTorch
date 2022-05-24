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

![torch pipeline](https://user-images.githubusercontent.com/86812576/170058296-9cb1ecce-e871-436a-ac7e-39c3b4b40e70.png)

Next, torchvision already has a default data pipeline.
data/train, after that there will be a data pipeline until finally the image size is according to our needs, which is 64x64, and this is only one image, later there will be a dataloader that automatically pulls only a total of 128 images (batch size). And because our images are colored so the total will be 128 images, 3 channels, 64 rows, and 64 columns or NCHW, this is the format on CNN
