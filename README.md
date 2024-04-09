# mood
**Abstract** : Imbalanced semi-supervised learning (SSL) has received significant attention due to the prevalence of class imbalanced and partially labeled data in real-world scenarios. Typically, a strict assumption is made that no out-of-distribution (OOD) data exists in the unlabeled data. However, in reality, naturally collected data inevitably contains samples that are outside of the target distribution. With OOD data, the performance of the imbalanced SSL methods drastically deteriorates, as blindly incorporating OOD data into model training will introduce noisy data. To address this issue, we propose an imbalanced SSL method called Mixup-OOD (MOOD). The core idea is to `turn waste into wealth', namely, exploring the potential of leveraging the seemingly detrimental OOD data to expand the feature space, particularly for tail classes, and achieve superior generalization performance on imbalanced data. Specifically, we first filter OOD data from the unlabeled data and then fuse it with labeled data using mixup to alleviate the lack of feature diversity, especially for tail classes. Based on OOD data mixup, we propose Push-and-Pull (PaP) loss to further expand the feature space of the tail classes, pulling in-distribution (ID) samples with OOD information towards the corresponding class centers while pushing outward away from OOD samples. Extensive experiments show that MOOD achieves superior performance compared with other state-of-the-art methods and exhibits robustness across data with different imbalanced ratios and OOD proportions.
#### Dependencies
- python 3.7.12
- PyTorch 1.8.1
- torchvision 0.9.1
- CUDA 11.1
- cuDNN 8.0.4
#### Dataset
- CIFAR-10
- CIFAR-100
- SVHN
#### Usage
Here is an example to run MOOD on CIFAR-10:

`python train.py --cfg cfg/cifar10_mood.yaml`
