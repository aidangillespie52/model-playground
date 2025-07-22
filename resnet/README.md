# ResNet

This project implements a **ResNet** from scratch in PyTorch and tests its ability to classify CIFAR-10 and some denoising tasks

**Reference Paper:**  
[Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)

**Notes**  
ResNet is basically convolutional blocks with skip connections. In the paper, it ends with flattening and a fully connected layer to output the class scores, which works since I'm just doing classification here.

The original paper doesn’t upsample at all, which makes sense for classification tasks, so I’ll just stick to that for now.  
I’ll probably do the paper implementation first, then later try out more modern resnet variations for other tasks like denoising or segmentation.

reading page four gives me the architecture we need for the blocks
went with 34 layer residual from paper