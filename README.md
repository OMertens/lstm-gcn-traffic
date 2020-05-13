# lstm-gcn-traffic
Code for the thesis of 'Applicability of Graph Convolutional Networks for Traffic Congestion Prediction' at KU Leuven.

Both models have everything in their file that is needed to run. All that is needed is the necessary the dataset and the necessary Python packages installed. For the GCN model, you will also have to get the CSV files.

Our code automatically checks for CUDA compatibility, otherwise training will happen on CPU.
The dataset is too large to post on GitHub, feel free to mail olivier.mertens@student.kuleuven.be to get the data via WeTransfer in the right format.

The code in graph-generation is what is used to create the CSV files in this repository and is not required to run the models.

Map.py is used to locate the detectors visually onto a map and is a also not required to run the models.
