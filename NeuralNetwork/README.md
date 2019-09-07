# Description
The training routine is implemented in the [Trainer.py](Trainer.py)
file, the network architecture is defined in the [Net.py](Net.py)
file and the code to load the data set is in the [DataLoader.py](DataLoader.py)
file. Besides the training code there is the [freeze_graph.bat](freeze_graph.bat)
which produces a ``frozen_graph.pb`` file with the definition of the 
network and the trained weights. This model is used to make live predictions 
on the smartphone.

# Dependencies
- python 3.6 or newer
- Tensorflow 1.14 and Matplotlib