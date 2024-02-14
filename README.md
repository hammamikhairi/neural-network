# A Neural Network from Scratch with Go (with Multiprocessing Support and Model Persistence)

This is a neural network library written entirely in Golang, developed for personal learning and experimentation purposes.

## Features

- Handcrafted from scratch.
- Implements multiprocessing for accelerated training.
- Achieves great results on various datasets:
  - MNIST Handwritten Digits Classification: 98% accuracy.
  - Wines Classification: 98.5% accuracy.
  - MNIST Fashion Dataset: 86% accuracy.
- Includes save and load support, allowing you to store trained neural networks for later use (potentially through an AP)I.

## Key Components

- Multi-layer Perceptrons (MLP)
- Backpropagation algorithm
- Stochastic Gradient Descent (SGD) optimizer
- Sigmoid, ReLU, Softmax, TanH and SiLU activation functions
- Cross-entropy, BinaryCrossEntropy and MeanSquareError loss functions

## Some more notes

- The model takes around 500 microseconds to make a prediction on a single core.
- You have the flexibility to customize the architecture of your neural network. You can specify the number of layers and the number of nodes in each layer to suit your specific needs.
- You can load your own data into the model. See [this example](https://github.com/hammamikhairi/neural-network/blob/master/Examples/LoadCustomData.go).
- This library is designed to be easily integrated into your own applications. Check out [this example](https://github.com/hammamikhairi/neural-network/blob/master/Examples/ServerIntergation.go).

## Web Interface

Explore and interact with the handwritten digits classifier on [this web interface](https://hammamikhairi.github.io/nn-front/)!

## Usage

You can get the library and test it locally by running :

```bash
go get github.com/hammamikhairi/neural-network
```

And you can view some examples [here](https://github.com/hammamikhairi/neural-network/tree/master/Examples)

## Acknowledgements

Special thanks to the open-source community for their contributions and support.
