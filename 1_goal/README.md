# Illustration

This subfolder contains the files used to create the first figure of the paper, which illustrates the idea of inference as well as our approach.

The figure is in the `fig/` subfolder as `.svg` and `.pdf` file.

`fig/fig1.graffle` is the OmniGraffle (macOS only) source file that contains the diagrams.

The data traces shown were generated using the notebook `01_make_traces.ipynb`.

The model needs compilation:

```
cd hodgkin_huxley  
python compile.py build_ext --inplace
```
