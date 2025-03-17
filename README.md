# Deep Neural Network Loss Function Development for Forecasting Extreme Events in Time Series Data
Xiao Xia Liang, Julien Straubhaar, Erwan Gloaguen, Maxime Claprood, and Philippe Renard

This paper presents a novel loss function, the extreme loss function, for DNN models solving regression-type problems. 
The extreme loss function is implemented in pytorch and tensorflow. 
Here, we tested the extreme loss function to forecast extreme events in karst spring discharges and groundwater heads in monitoring wells. 
The 2 models selected to test the extreme loss function are the [Graph WaveNet](https://arxiv.org/pdf/1906.00121) ([code](https://github.com/sshleifer/Graph-WaveNet))and the [GRU auto-encoder](https://arxiv.org/pdf/1406.1078).

The Graph WaveNet model has been adapted for hydrogeological data. The model is developed with pytorch.
The GRU auto-encoder is developed with tensorflow-keras.

### Package requirements

- python 3
- pandas (2.1.1)
- numpy (1.26.4)
- scikit-learn (1.4.0)
- scipy (1.13.1)
- torch (2.1.2+cu121)
- tensorflow (2.14.0)
