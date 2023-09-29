# DFGP-ICCV-2023

A repository of **'[Data Augmented Flatness-aware Gradient Projection for Continual Learning. ICCV, 2023.](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Data_Augmented_Flatness-aware_Gradient_Projection_for_Continual_Learning_ICCV_2023_paper.pdf)'**.

## Abstract
> The goal of continual learning (CL) is to continuously learn new tasks without forgetting previously learned old tasks. To alleviate catastrophic forgetting, gradient projection based CL methods require that the gradient updates of new tasks are orthogonal to the subspace spanned by old tasks. This limits the learning process and leads to poor performance on the new task due to the projection constraint being too strong. In this paper, we first revisit the gradient projection method from the perspective of flatness of loss surface, and find that unflatness of the loss surface leads to catastrophic forgetting of the old tasks when the projection constraint is reduced to improve the performance of new tasks. Based on our findings, we propose a Data Augmented Flatness-aware Gradient Projection (DFGP) method to solve the problem, which consists of three modules: data and weight perturbation, flatness-aware optimization, and gradient projection. Specifically, we first perform a flatness-aware perturbation on the task data and current weights to find the case that makes the task loss worst. Next, flatness-aware optimization optimizes both the loss and the flatness of the loss surface on raw and worst-case perturbed data to obtain a flatness-aware gradient. Finally, gradient projection updates the network with the flatness-aware gradient along directions orthogonal to the subspace of the old tasks. Extensive experiments on four datasets show that our method improves the flatness of loss surface and the performance of new tasks, and achieves state-of-the-art (SOTA) performance in the average accuracy of all tasks.


## Citation
If you find our paper or this resource helpful, please consider cite:
```
@InProceedings{DFGP_ICCV_2023,
    author    = {Yang, Enneng and Shen, Li and Wang, Zhenyi and Liu, Shiwei and Guo, Guibing and Wang, Xingwei},
    title     = {Data Augmented Flatness-aware Gradient Projection for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {5630-5639}
}
```
Thanks!


## Code

Please configure the path of the data set in dataloader first.

Run PMNIST
> python main_dfgp_pmnist.py

Run CIFAR100
> python main_dfgp_cifar100.py

Run Five Datasets
> python main_dfgp_fivedataset.py

Run MiniImagenet
> python main_dfgp_miniimagenet.py


Tip: The default hyperparameters in the main_dfgp_xxx.py file are not necessarily the optimal hyperparameters. You can further check the hyperparameter configuration in our **./logs/xxx/log_date.txt** to reproduce the results.


## Acknowledgement
Our implementation references the code below, thanks to them.

[sahagobinda/GPM](https://github.com/sahagobinda/GPM), [davda54/SAM](https://github.com/davda54/sam)
