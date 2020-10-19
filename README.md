# Training deep neural density estimators to identify mechanistic models of neural dynamics

Code for [**Training deep neural density estimators to identify mechanistic models of neural dynamics**](https://elifesciences.org/articles/56261). Each application has its own subfolder.

The experiments in the paper were run using SNPE based on Theano, using the [`delfi` toolbox](http://www.mackelab.org/delfi/) as installed below. For new applications of SNPE, we recommend using the [`sbi` toolbox](http://www.mackelab.org/sbi/), which is based on PyTorch and has extended functionality.


## Base environment

Setup a base environment for running these experiments as follows:

```commandline
conda create -n ind python=3.7
conda activate ind
conda install numpy scipy matplotlib ipython jupyter jupyterlab pandas seaborn

pip install dill python-box svgutils cython
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install theano --upgrade
pip install parameters cma tqdm dill==0.2.7.1 python-box==3.1.1

git clone https://github.com/mackelab/delfi.git
cd delfi
pip install -r requirements.txt
pip install .

ipython kernel install --name "ind" --user
```

Use the `ind` kernel when running notebooks. If applications have extra dependencies or require compilation of Cython models, this is stated in the `README.md` of the application subfolder.


Make sure to use float32 precision with theano, e.g. by adding this to the theano config file (`~/.theanorc`):

```commandline
[global]
floatX = float32
```


## Citation

```bibtex
@article{gonccalves2020training,
  title     = {Training deep neural density estimators to identify mechanistic models of neural dynamics},
  author    = {Gon{\c{c}}alves, Pedro J and Lueckmann, Jan-Matthis and Deistler, Michael and Nonnenmacher, Marcel and {\"O}cal, Kaan and Bassetto, Giacomo and Chintaluri, Chaitanya and Podlaski, William F and Haddad, Sara A and Vogels, Tim P and Greenberg, David S. and Macke, Jakob H.},
  year      = {2020},
  doi       = {10.7554/eLife.56261},
  publisher = {eLife},
  journal   = {eLife}
}
```

## License

MIT
