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

So that every time you pull a minibatch of 128 images, it's a **_never-before-seen_** image because it's augmented. For example, the augmentation of rotation is 15 degrees, after that it is randomly resized by 64 cropping, then it will be randomly flipped horizontally or vertically, and finally converted to a tensor. So data augmentation just plugs in the pipeline, and when the DataLoader wants to pull data, it will process the argumentation.

Don't augment test data, because it's predictable, don't do it. only do crop_size so that the size fits the input because the number of features is already fit.
The trick is to resize to 70, then CenterCrop to 64. why do you need to resize? because the image size is different depending on the device in which the image is taken.

### Sanity Check
let's check if it works well.

![Screenshot 2022-05-24 214726](https://user-images.githubusercontent.com/86812576/170064737-1b8d3b87-7d66-4419-ae23-7ffdbca82f6f.png)

feature shape is appropriate. With two classes, namely 'invasive' and 'noninvasive'.

# Architecture and Config
## Architecture
In this project I use the architecture as below:

![arsitektur](https://user-images.githubusercontent.com/86812576/170066662-67b38b46-b407-4403-8daa-92b20a91ff9e.png)

The input is 3 channels with 64x64 pixels. later this architecture will perform convolution-activation-pooling 4 times, and the activations used are ReLU (rectify linear unit) and Max Pooling.

Convolution (3, 8) means that it uses 3 RGB channels and becomes 8 channels, which means that it uses 8 filters. The 8 channels in question are feature maps that are concatenated so that it looks like 8 channels. from 8 channels I use 16 filters, so it becomes 16 channels. to make it easier just imagine it's like a neural network, it's like a neuron but actually it's a filter. 

In the Neural Network, the neurons in the architecture are getting less and less, in contrast to CNN, which is getting more and more complex. After reaching the end and doing 4 times pooling the image size is halved. every time you do Conv-ReLu-MaxPool the image size becomes smaller because of MaxPool, in the end there are only 16 features (4x4) and 64 channels, so the total feature is 1024 channels (64x4x4) which is lighter than ANN.

![feature extr](https://user-images.githubusercontent.com/86812576/170180780-ffa92b55-a4e5-4741-a56a-d436e15b2444.png)

After flattening enter it into the neural network. so there are 2 phases. The first is the feature extractor, because when there is an image, the features are extracted until they become flattened.

![fully con](https://user-images.githubusercontent.com/86812576/170180802-e0b49edb-7b1e-4e39-9301-d80afc74670a.png)

The second phase is after being flattened, enter it into a fully connected neural network. So there are always 2 phases on CNN, namely feature extractor and fully connected. Linear, you can directly enter the amount according to the features that have been flattened and then immediately become 2 classes, namely invasive and noninvasive. But it will be made gradually from 1024 to 256, then into 2 features, you can use _logsoftmax_ and the loss is _NLL_.

here is the custom wrapped code.

![buia](https://user-images.githubusercontent.com/86812576/170181637-5ea209d5-262f-4997-aa4a-216509648250.png)

The way the code above works is that it will first perform a feature extraction, then enter it into fully connected where there is dropout = 0.1, and activation = 'lsoftmax' as explained earlier.

## Config
On config will save only batch_size and crop_size. while the architecture is not saved, neither is the input channel as it changes frequently depending on the case.

# Training Preparation -> MCOC
![mcoc](https://user-images.githubusercontent.com/86812576/170183931-96bf4d90-bd01-4b40-97b8-d97d8869acd3.png)

- Model

Contains a built-in model that exists on an architecture called CustomCNN.

- Criterion

cause activation is logsoftmax then use NLLLoss()

- Optimizer

In the Optimizer, I use AdamW, namely Adaptive Momentum with Weight Decay, where there is a regulation to reduce overfit. With learning rate 0.001

- Callback

The callback will save every 50 epochs of progress, and will plot the cost error every 20 epochs, and will save it in a folder.

### Callback
Callback is something called repeatedly. Callback for common PyTorch Workflow:
- Logging

Automatically will plotting and logging.

- Checkpoint

To save epoch progress.

- Runtime Plotting
To monitor the loss in progress, and will plot the score.

- Early Stopping

For example, if the test cost has started to increase, it will stop and take the best cost.
If reached early stopping, the training will be stopped.

# Training and Result

![tnr](https://user-images.githubusercontent.com/86812576/170208446-87d6cbc7-b5ba-4a22-b890-cacde903f9f6.png)

In training, I limit the early stopping to 5, and after touching the early stopping the best model is **0.8225**. can be seen that is not very overfit. The interesting thing is that the test is lower than the train, meaning that the test is better than the train, maybe because it does augmentation and also adds dropouts.

One important thing is why can test is better than train? In deep learning we will often meet where testing is better than training in accuracy or loss. the answer is because the sequence of doing the loop, first when train on feedforward, calculate loss, backprop, and update weight, train accuracy and cost accuracy are calculated during training and then averaged, but after updating the new weights then calculating the cost from the test means that the test will be a little smarter than the train. So as if test is better than train.

# Predict
For prediction, I just pull a batch of data as an example.
