## Knowledge Graph Embeddings with Keras

This repository contains implementations of popular Knowledge Graph Embedding models in `tensorflow-keras`.

Currently the following models are included:
* ComplEx
* DistMult
* Rescal
* Multi-MLP

#### Installation
The package `keras-kge` can be installed using `pip`:

```
git clone https://github.com/baierst/keras-kge.git
cd keras-kge
pip install .
```

####Example

An example using the popular kinship graph dataset, can be found in the `examples` folder. Run the example as follows:

```
cd examples/
python kinship.py
```