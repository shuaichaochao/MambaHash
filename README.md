# MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval（ICMR 2025）

## The Overall Architecture Of MambaHash
![figure2](https://github.com/user-attachments/assets/70f4b93c-e0bd-47c5-be48-4a4081227c0e)
The detailed architecture of the proposed MambaHash. MambaHash accepts pairwise images as input, and adopts a similar stem architecture to divide the images into overlapping patches with the generated patches fed into the Mamba block. The whole model architecture consists of four stages, followed by an Adaptive feature enhancement module to increase feature diversity. Finally, the binary codes are output after the hashing layer.
