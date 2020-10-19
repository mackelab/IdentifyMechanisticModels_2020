# Channelomics

## Requirements

### Model

The model needs compilation:

```
cd model  
python compile.py build_ext --inplace
```


### NEURON

Because simulation results for the channels in IcG (IonChannelGenealogy) are downsampled, we re-ran their protocols and stored output without downsampling (for the channels we are interested in). This part requires a local NEURON installation; see instructions for [Linux](https://neuron.yale.edu/neuron/download#linux) and [macOS](https://neuron.yale.edu/neuron/download/compilestd_osx).


## Reproducing results and figures

Run:
```
jupyter notebook
```
in the environment and go through the notebooks in numerical order.
