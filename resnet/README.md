# ResNet

This project implements a **ResNet** from scratch in PyTorch and tests its ability to classify CIFAR-10 and some denoising tasks

**Reference Paper:**  
[Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)

**Notes**  
ResNet is basically convolutional blocks with skip connections.  
In the paper, it ends with flattening and a fully connected layer to output the class scores,  
which works since I'm just doing classification here.

This makes sense for classification tasks, so I’ll just stick to that for now.

Reading page 4 of the paper gives me the architecture needed for the blocks.  
I'm going with the **34-layer residual network** from the paper.
