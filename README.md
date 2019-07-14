# PortHamiltonianNN
Code for "Port-Hamiltonian Approach to Neural Network Training" submitted to 58th IEEE Conference on Decision and Control (CDC 2019)

### Content:

```pyPH/model.py``` contains the new optimizer class proposed in the paper. PHNNs can take as input nn.Modules and provides a fit method.

```pyPH/numpy_simple.py``` contains a numpy implementation of a single linear predictor along with functions that describe the Port-Hamiltonian ODE of its parameters. For general use import the PHNN class in ```pyPH/model.py``` instead.
