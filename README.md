# ml-dl-projects

This repository contains a series of Jupyter notebooks that demonstrate various applied AI concepts. The projects cover machine learning, deep learning, and building a neural network from scratch as part of my minor in applied AI.

We were instructed to cover a wide range of topics in AI, encompassing approximately 50 different areas. However, the implementation of these projects was left open-ended, allowing for creative freedom in their development, meaning that the projects below are my own ideas and implementations.

## Projects

### Machine Learning

- [Predicting Life Expectancy using Linear Regression](./ml/life_expectancy-linear_regression.ipynb) - A linear regression model to predict life expectancy based on a dataset from the World Health Organization.
- [Binary Classification of Organic-Non-Organic Images](./ml/waste-binary_classification.ipynb) - A binary classification model to classify images of organic and non-organic waste.
- [Songe Genre Multi-Class Classfication](./ml/song_genre-multi_class_classification.ipynb) - Trying to classify a collection of 30000 spotify songs into 6 genres.
- [Glass Identification with PCA](./ml/glass_identification-pca.ipynb) - Exploring the UCI Glass Identification dataset using PCA.
- [Clustering of Wine Data](./ml/wine-clustering.ipynb) - Clustering the UCI Wine dataset using KMeans and Agglomerative clustering.

### Deep Learning

- [Game Strategy using Reinforcement and Transfer Learning](./dl/game_strategy-reinforcement+transfer_learning.ipynb) - Custom GridWorld implementation that uses reinforcement learning to train an agent to and transfer learning for scaling to a larger grid.
- [Text Classfication using an RNN and BERT](./dl/text_classification-rnn-transformers.ipynb) - Classifying tripadvisor hotel reviews using an RNN and a pre-trained BERT model.
- [Anomaly Detection using Autoencoders and Adversarial Training](./dl/anomaly_detection-autoencoders_adversarial-learning.ipynb) - Detecting anomalies in the MNIST dataset using autoencoders and improving the model robustness using adversarial training.
- [Fruit Classification using DNN, CNN, and ResNet50](./dl/fruit_classifcation-dnn-cnn-resnet.ipynb) - Classifying images of fruits using a Deep Neural Network, Convolutional Neural Network, and ResNet50.

### Neural Network From Scratch

- [Building a Neural Network from Scratch](./nn_from_scratch/nn_from_scratch.ipynb) - Building a neural network from scratch using only numpy and implementing forward and backpropagation.

## Setup

To run the notebooks locally (though they can be viewed directly on GitHub), follow these steps:

1. Clone the repository:

```bash
    git clone https://github.com/yourusername/ml-dl-projects.git
    cd ml-dl-projects
```

2. Create a virtual environment (Optional but recommended as per [PEP 405](https://peps.python.org/pep-0405/)):

```bash
    python -m venv venv
    source env/bin/activate  # On Windows, use `.\env\Scripts\activate`
```

3. Install the required packages:

```bash
    pip install -r requirements.txt
```

4. Start Jupyter:

```bash
    jupyter notebook
```
