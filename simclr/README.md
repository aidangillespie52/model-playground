# SimCLR (Simple Contrastive Learning of Representations)

This project implements **SimCLR**, a self-supervised learning method that trains an encoder to produce meaningful image representations without labels, by maximizing agreement between differently augmented views of the same image.

**Goal:**  
Train a SimCLR model on the **CIFAR-10** dataset to learn robust feature embeddings through contrastive learning, enabling better downstream performance on classification tasks.

**Reference Paper:**  
[A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)

**Notes**  
There are many different ways to do the augmentation, and it would probably be best if I do more research on them;
however, I just want something to test rn and then I'll look into different augmentations

Seems like it's just built on resnets, so before i start this i want to understand resnets and i'll come back when i understand it
