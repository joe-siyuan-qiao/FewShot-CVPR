## Get Started
### Data Preparation
The dataset we use here is MiniImageNet, which is a subset of the large-scale ImageNet-1K dataset.
We use the data split shown in the folder './data/split.json'.
The category split between the large-scale part and the few-shot part is the same as that of [Meta-Learner LSTM](https://openreview.net/pdf?id=rJY0-Kcll).
Please download the ImageNet dataset from [here](www.image-net.org/) and put them in the folder './data' according to './data/split.json'.
After the preparation,
there should be three folders in './data': './data/train' for holding the training data of the large-scale part, './data/val' for the large-scale validation, and the few-shot './data/test' that our task mostly cares about.

### Large-Scale Training
We first train a convolutional model on the large-scale part of the MiniImageNet dataset as our activation function.
We use a popular training framework [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
For training,
```bash
cd train_large
bash run.sh
```
After training, the model can be found in './few_shot/pretrained/model.t7'.
By default, we use a shallow convolutional network similar to [Matching Network](https://arxiv.org/abs/1606.04080).
But it also includes the Wide-ResNet model adapted to MiniImageNet.

### Few-Shot Training and Evaluation
Next, we train and evaluate our few-shot performances on the few-shot part of the MiniImageNet dataset.
To do this,
```bash
cd few_shot
bash run.sh
```
The final two lines are the means and standard derivations of the one-shot and 5-shot accuracies in the 5-way test.

There are also some tricks for getting better performances that we do not present in the codes nor in the paper.
1. It is easy to see that the few-shot performance can be further improved if the activation is more _average_.
One way to make them look more _average_ is by cropping each example multiple times at different positions then using the mean activations.
We change the center-crop to ten-crop in './few_shot/ftext-val.lua' and report the multi-view performance.
2. Another way to make the activation more _average_ is by using another network to first map the individual activation to the mean activation, then train another network to convert the mean activation to the parameters.
We did not implement this on the MiniImageNet dataset.

## Performances
| Method        | 1-Shot        | 5-Shot  |
| ------------- |:-------------:| -----:|
| Matching Network | 43.56 (0.84) | 55.31 (0.73%) |
| Meta-Learner LSTM      |  43.44 (0.77) |   60.60 (0.71)|
| MAML |  48.70 (1.84)      |    63.11 (0.92)|
| Ours-Simple (in paper) | 54.53 (0.40) | 67.87 (0.20) |
| Ours-Simple (multi-view) | 56.01 (0.59) | 70.70 (0.44) |
| Ours-WRN (in paper) | 59.06 (0.41) | 73.74 (0.19) |
