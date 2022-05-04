# ai

The **AI** repository is a place to gather important research on the topic. Here you will find useful code, datasets, and links to research papers. All scoreboard results shown are from experiments I have run myself. 



## Scoreboards



| Dataset                                                      | Metrics              | Task                 | Notebook                                                     | Model                           | Frameworks        |
| ------------------------------------------------------------ | -------------------- | -------------------- | ------------------------------------------------------------ | ------------------------------- | ----------------- |
| [mnist](https://www.tensorflow.org/datasets/catalog/mnist)   | 99.25% Test Accuracy | Image classification | [simple_mnist_keras_covnet.ipynb](./image-datasets/mnist/simple_mnist_keras_covnet.ipynb) | simple 4 layer CNN with dropout | Keras, Tensorflow |
| [imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 85.61% Test Accuracy | Sentiment analysis   | [sentiment-analysis-using-lstm-pytorch.ipynb](./nlp-datasets/imdb/sentiment-analysis-using-lstm-pytorch.ipynb) | LSTM                            | PyTorch           |
| [deskdrop](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop) | Recall@5 of 0.2417   | Recommendation       | [recommender-system.ipynb](./recommender-datasets/deskdrop/recommender-system.ipynb) | Popularity Model                |                   |
| [deskdrop](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop) | Recall@5 of 0.1628   | Recommendation       | [recommender-system.ipynb](./recommender-datasets/deskdrop/recommender-system.ipynb) | Content-Based Filtering Model   |                   |



## Datasets



* **Image Datasets**
  * ImageNet Dataset
  * MNIST Dataset [source](http://yann.lecun.com/exdb/mnist/)
  * Fashion MNIST Dataset
  * Coco Dataset [source](https://cocodataset.org/#home)
  * CIFAR-10
  * CIFAR-100
* **Tabular Datasets**
  * Boston Housing Dataset
  * Iris Flow Dataset
  * Breast Cancer Wisconsin (Diagnostic) Dataset [source](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
* **NLP**
  * IMDB Dataset of 50K Movie Reviews [source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  * Twitter Sentiment Analysis Dataset 
* **Recommender System Datasets**
  * CI&T DeskDrop [source](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop)



## Research Papers



### My Research Papers

* [Illuminating Diverse Neural Cellular Automata for Level Generation](https://arxiv.org/abs/2109.05489)
  * Selected for presentation at the 2022 Genetic and Evolutionary Computation Conference



### Important Research

* 3D Points
  * [PointCNN: Convolution on X-Transformed Points](https://arxiv.org/abs/1801.07791)



## Terms



**Recall**

* Answers the question: "How many relevant items are retrieved?"
* $\text{recall}=\frac{\text{true positives}}{\text{false negatives}+\text{true positives}}$

