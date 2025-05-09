# Copmuter Vision - Image Classification
Comparative study of lightweight versions of classic CNN architectures and a lightweight Vision Transformer on typical image classification datasets (MNIST and Cifar).

## Project Outline
The flow of this project will be as follows: (1) Implement model architectures listed down below. (2) Train models on one or more of the datasets listed below (original, agumented and MixUp versions) to assess architecture performance in different data preprocessing contexts. 

## Model Architectures (Lite Versions)
| MODEL      | PAPER                    | COMPANY   | IMPL         |
|------------|--------------------------|-----------|--------------|
| LeNet      | LeCun et al., 1998       | -         | $\checkmark$ |
| ResNet     | He et al., 2015          | Microsoft | $\checkmark$ |
| DenseNet   | Huang et al., 2018       | Facebook  | $\checkmark$ |
| MobileNet  | Howard et al., 2017      | Google    | $\checkmark$ |
| ViT        | Dosovitskiy et al., 2021 | Google    | $\checkmark$ |
| GoogLeNet  | Szegedy et al., 2014     | Google    | -            |
| SqueezeNet | Iandola et al., 2016     | -         | -            |

## Datasets
| DATASET      | SHAPE               | SIZE           | INFO 
|--------------|---------------------|----------------|-----------------------------------
| MNIST Digits | 1x28x28 (Grayscale) | 70K (60/10) | Handwritten digits 0-9, 10 classes
| FashionMNIST | 1x28x28 (Grayscale) | 70K (60/10) | Zalando clothes, 10 classes
| Cifar-10     | 3x32x32 (RGB)       | ...?           | Distinct image classes, 10 classes
| Cifar-100    | 3x32x32 (RGB)       | ...?           | Distinct image classes, 100 classes
| Other ? | ...? | ...? | ...?  

## Training
* Original | No data augmentation
* Augmented:
    - Flips
    - Rotations
    - Crops
    - [Distortion] - omitted pixels
    - [Glare] - areas of highlights and fasded contour
    - [Scratches] - lines of removed pixels
* MixUp | Overlapped images and one-hot labels
    - E.g. $\lambda \times \text{Im}_1 + (1-\lambda) \times \text{Im}_2$, same for one-hot labels
    - Idea : To make network generalise better

## Evaluation
* Loss
<!-- * Error rate -->
* Accuracy
* Class Precision, Recall & F1-Score
* Training Time / Epoch
* Inference Time / Epoch
* Parameter Count
* FLOPs
* Training Time / Parameter
* Training Time / FLOP






## E. Clothing Classification
### 3. Comparing the performance of different network architectures

Fashion-MNIST is a dataset of Zalando's article images-consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. One can design a convolutional neural network or Transformer to address the classification problem.




# SC4001 CE: Neural Networks and Deep Learning

## Project Requirements
Students are to propose and execute a final project on an application or a research issue that is related to neural networks and deep learning. The project can be carried out in a group consisting of no more than three members. Students are to come up with a potential technique for the application or to mitigate the issue, to develop associated codes, and to compare with existing methods. Students may choose, focus, and expand on the project ideas A – F given below.

By the deadline, students are to submit a project report in a .pdf file of ten A4 pages (Arial 10 font) and associated code in a .zip file to NTULearn.

The project report should have the names of the team members on the front page and contain an
introduction to the project idea, a review of existing techniques, a description of the methods used, experiments and results, and a discussion. The 10-page limit is exclusive of references, content page, and cover page. The code needs to be commented properly. Make sure the code can be tested easily.

The assessment is based on the project execution (30%), experiments and results (30%), report
presentation (15%), and novelty (15%), and peer review (10%. Conducted via Eureka). We apply the same late submission penalty as in Assignment 1, i.e., 5% for each day up to three days.
