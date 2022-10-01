# ai

The **AI** repository is a place to gather important research on the topic. Here you will find useful code, datasets, and links to research papers. All scoreboard results shown are from experiments I have run myself. 



## Scoreboards



| Dataset                                                      | Metrics              | Task                 | Notebook                                                     | Model                                   | Frameworks        |
| ------------------------------------------------------------ | -------------------- | -------------------- | ------------------------------------------------------------ | --------------------------------------- | ----------------- |
| [mnist](https://www.tensorflow.org/datasets/catalog/mnist)   | 99.25% Test Accuracy | Image classification | [simple_mnist_keras_covnet.ipynb](./image-datasets/mnist/simple_mnist_keras_covnet.ipynb) | simple 4 layer CNN with dropout         | Keras, Tensorflow |
| [imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 85.61% Test Accuracy | Sentiment analysis   | [sentiment-analysis-using-lstm-pytorch.ipynb](./nlp-datasets/imdb/sentiment-analysis-using-lstm-pytorch.ipynb) | LSTM                                    | PyTorch           |
| [deskdrop](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop) | Recall@5 of 0.3426   | Recommendation       | [recommender-system.ipynb](./recommender-datasets/deskdrop/recommender-system.ipynb) | Hybrid: Collaborative and Content-Based | Numpy, Pandas     |



## Examples

[face_detection.ipynb](./examples/face_detection.ipynb): A Jupyter notebook demonstrating four different face detection strategies. Run and tested using Google Colab. 



## Datasets



### Dataset Sources

* [Scikit Learn Real world datasets](https://scikit-learn.org/stable/datasets/real_world.html)

* [TensorFlow Datasets](https://www.tensorflow.org/datasets)
* [PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [Google Dataset Search](https://datasetsearch.research.google.com/)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* [Papers with Code](https://paperswithcode.com/datasets)
* [OpenML](https://www.openml.org/)
* [Hugging Face Datasets](https://huggingface.co/datasets)





### Datasets by Category

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



## Machine Learning Examples

* [Papers With Code](https://paperswithcode.com/)
* [Hugging Face](https://huggingface.co/)
* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Spark ML Guide](https://spark.apache.org/docs/1.2.2/ml-guide.html)
* [Keras Code Examples](https://keras.io/examples/)



## Model Sources

* [TensorFlow Hub](https://tfhub.dev/)
* [Hugging Face Models](https://huggingface.co/models)
* [PyTorch Hub](https://pytorch.org/hub/)
* [PyTorch Zoo](https://pytorch.org/serve/model_zoo.html)





## Research Papers



### My Research Papers

* [Illuminating Diverse Neural Cellular Automata for Level Generation](https://arxiv.org/abs/2109.05489)
  * Selected for presentation at the 2022 Genetic and Evolutionary Computation Conference



### Important Research

* 3D Points
  * [PointCNN: Convolution on X-Transformed Points](https://arxiv.org/abs/1801.07791)
* Face Recognition
  * [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition](https://paperswithcode.com/paper/ms-celeb-1m-a-dataset-and-benchmark-for-large)
  * [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://paperswithcode.com/paper/facenet-a-unified-embedding-for-face)
  * [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep)




## Frameworks

* [TensorFlow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)
* [MXNet](https://mxnet.apache.org/versions/1.9.1/)
* [Caffe](https://caffe.berkeleyvision.org/)
* [OpenCV](https://opencv.org/)
* [scikit-learn](https://scikit-learn.org/stable/index.html)



## Terms



**Recall**

* Answers the question: "How many relevant items are retrieved?"
* $\text{recall}=\frac{\text{true positives}}{\text{false negatives}+\text{true positives}}$

