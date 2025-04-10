# GAUDI
By Mirja Granfors, Jesús Pineda, Blanca Zufiria Gerbolés, Joana B. Pereira, Carlo Manzo, and Giovanni Volpe.

GAUDI is an unsupervised geometric deep learning framework for analyzing complex graph-structured data. GAUDI's hourglass architecture, with multiple hierarchical pooling and upsampling steps, maps graphs into a structured latent space, capturing their underlying parameters.

This repository contains an implementation of GAUDI, introduced in [Global graph features unveiled by unsupervised geometric deep learning](https://arxiv.org/abs/2503.05560).

## Getting started

An example of how GAUDI is trained on Watts-Strogatz small-world graphs can be found here:

[Training GAUDI on Watts-Strogatz Small-World Graphs](GAUDI_Watts_Strogatz.ipynb)

## Dependencies

To use this implementation, ensure you have the following dependencies installed:

- `deeplay`
- `torch`

You can install them using:

```bash
pip install deeplay
```
```bash
pip install torch
```

