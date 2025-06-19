# MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval（ICMR 2025）

## The Overall Architecture Of MambaHash
![figure2](https://github.com/user-attachments/assets/382d08d5-0618-4978-8a60-0f867a40039c)
Figure.1. The detailed architecture of the proposed MambaHash. MambaHash accepts pairwise images as input, and adopts a similar stem architecture to divide the images into overlapping patches with the generated patches fed into the Mamba block. The whole model architecture consists of four stages, followed by an Adaptive feature enhancement module to increase feature diversity. Finally, the binary codes are output after the hashing layer.

![image](https://github.com/user-attachments/assets/eaa07f7e-db3c-4f1f-a312-3eb5135d27f3)

We further investigate the mainstream backbone networks utilized by existing deep hashing methods, including convolutional neural networks (exemplified by ResNet50), Transformer (exemplified by ViT-L/32), and hybrid networks (exemplified by HybirdHash).
![image](https://github.com/user-attachments/assets/ed7c5a73-c76e-412b-91bc-6f18ecb09acc)
![image](https://github.com/user-attachments/assets/0d08e97b-f6b6-4da8-8a10-6d6cff5841a4)

##Notice

We utilized partial pre-trained model weights, named groupmamba_small_ema.pth, with the link [Amshaker](https://github.com/Amshaker/GroupMamba)
## Datasets

The following references are also derived from a [swuxyj](https://github.com/swuxyj/DeepHash-pytorch)

There are three different configurations for cifar10

   * config["dataset"]="cifar10" will use 1000 images (100 images per class) as the query set, 5000 images( 500 images per class) as training set , the remaining 54,000 images are used as database.
    
   * config["dataset"]="cifar10-1" will use 1000 images (100 images per class) as the query set, the remaining 59,000 images are used as database, 5000 images( 500 images per class) are randomly sampled from the database as training set.
    
   * config["dataset"]="cifar10-2" will use 10000 images (1000 images per class) as the query set, 50000 images( 5000 images per class) as training set and database.

You can download NUS-WIDE [here](https://github.com/swuxyj/DeepHash-pytorch)

Use data/nus-wide/code.py to randomly select 100 images per class as the query set (2,100 images in total). The remaining images are used as the database set, from which we randomly sample 500 images per class as the training set (10, 500 images in total).

You can download ImageNet, NUS-WIDE-m and COCO dataset [here](https://github.com/swuxyj/DeepHash-pytorch) where is the data split copy from

NUS-WIDE-m is different from NUS-WIDE, so i made a distinction.

269,648 images in NUS-WIDE , and 195834 images which are associated with 21 most frequent concepts.

NUS-WIDE-m has 223,496 images which are associated with 81 concepts, and NUS-WIDE-m is used in HashNet(ICCV2017). Of these, removing the incorrectly labeled images, there are 173692 images, including 5000 images for the test set and the rest for the retrieval set.
