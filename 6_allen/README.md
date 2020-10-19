# Hodgkin-Huxley on Allen data

## Extra requirements

### Python

Install the following in your base environment:

```
conda install pytables
pip install allensdk
```

### Model

The model needs compilation:

```
cd model  
python compile.py build_ext --inplace
```


### Data

Data has been pre-downloaded from Allen Cell Type Database, pre-processed, and are in folder (`/support_files`). This has been done by running `/support_files/extract_AllenDB_python2.py` with python 2.7.


## Reproducing results and figures

Run:
```
jupyter notebook
```
in the environment and go through the notebooks in numerical order.
