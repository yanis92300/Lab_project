# Lab-project : using Segment Anything (SAM) to segment cancer cells in H&E Images

This repository contains the code implementation of our lab project which consists of investigating how [SAM](https://github.com/facebookresearch/segment-anything.git) can be leveraged for automatic cancer cell detection in H&E images.

Specifically, our contribution is two-fold:

1. We show how te performances of SOTA model [Hovernet](https://github.com/vqdang/hover_net.git) can be improved with SAM used as a post-processing
2. We combine the efforts made in the works of [MedSAM](https://github.com/bowang-lab/MedSAM.git) and [CellViT](https://github.com/TIO-IKIM/CellViT.git) to slightly improve the SOTA performances in automated instance segmentation of cell nuclei in digitized tissue samples. More specifically, we use the weights of the ViT encoder from MedSAM in the training of CellViT.

Both experiments are made on the [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) dataset, a challenging nuclei instance segmentation benchmark.
