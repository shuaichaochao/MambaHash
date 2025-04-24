# MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval（ICMR 2025）

## The Overall Architecture Of MambaHash
![figure2.pdf](https://github.com/user-attachments/files/19890406/figure2.pdf)
Figure.1. The detailed architecture of the proposed HybridHash. We adopt similar segmentation as ViT to divide the image with finer granularity and feed the generated image patches into the Transformer Block. The whole hybrid network consists of three stages to gradually decrease the resolution and increase the channel dimension. Interaction modules followed by each stage to promote the communication of information about the image blocks. Finally, the binary codes are output after the hash layer.
