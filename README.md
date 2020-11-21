# DeepKinFam-interpretable-DNN

## Overview 

This Python script is used to train and test deep neural networks model for prediction of kinase family inhibitors (KFI), and uses the DeepSHAP algorithm to calculate the contribution of each compound feature in the model to facilitate model interpretation. Deep neural networks model will be built by Keras with tensorflow, and use [NVIDIA](https://developer.nvidia.com/) GPU for calculations. You can set almost model hyper-parameters as you want. Compound data must be written as csv file format (see Data Specification for details). 

!["DeepKinFam-interpretable-DNN"](https://github.com/EthanFY/DeepKinFam-interpretable-DNN/blob/main/docs/artwork/DNN_structure.png?raw=true "DeepKinFam-interpretable-DNN")

### Table of contents:
- [Requirements](#Requirements)
- [Data Specification](#Data-Specification)
- [Reference](#Reference)
- [Contact](#Contact)

## Requirements

You can install from either PyPI or conda-forge. Here used [Anaconda](https://www.anaconda.com/) as the platform for operation and management.

- [tensorflow-gpu](https://www.tensorflow.org/install/gpu) > 1.14  
`conda install -c anaconda tensorflow-gpu`  
- [keras-gpu](https://www.tensorflow.org/install/gpu) > 2.2  
`conda install keras-gpu`  
- [numpy](https://anaconda.org/conda-forge/numpy)  
`conda install -c conda-forge numpy`  
- [pandas](https://anaconda.org/conda-forge/pandas)  
`conda install -c conda-forge pandas`  
- [scikit-learn](https://anaconda.org/conda-forge/scikit-learn)  
`conda install -c conda-forge scikit-learn`  
- [shap](https://github.com/slundberg/shap)  
`conda install -c conda-forge shap`
- [cudnn](https://developer.nvidia.com/rdp/cudnn-archive) (Please install the version corresponding to the GPU specification)  
- [cudatoolkit](https://developer.nvidia.com/cuda-toolkit) (Please install the version corresponding to the GPU specification)  



## Data Specification
The input data format is a `.csv` file, and the active and inactive compounds that have been statistically filtered for the kinase fimaly. The data is composed of the following table, the first and second columns are `Inchikey` and `Label`, label `0` as negative and `1` as positive. Columns 2 to 240 show that the 238 `features` of the compound consist of [Checkmol fingerprints](https://homepage.univie.ac.at/norbert.haider/cheminf/cmmm.html) and Drug moieties. Binary encoding of the compounds, with this feature, give `1`, otherwise it will be `0`.

|  Inchikey  |  Label  |  f-1  | ... | f-238 |
| :-----------:|:---------:|:-------:|:-----:|:-------:|
| AAGKMGNYUYCEPD-UHFFFAOYSA-N  |  1  |  0    | ... |  0  |
|  ...                         | ... | ...   | ... | ... |
| ZZPJUIRSEQUVBV-UHFFFAOYSA-N  |  0  |  0    | ... |  0  |


## Reference
1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436-444.
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems (pp. 4765-4774).
3. Haider, N. (2010). Functionality pattern matching as an efficient complementary structure/reaction search tool: an open-source approach. Molecules, 15(8), 5079-5092.
4. Merget, B., Turk, S., Eid, S., Rippmann, F., & Fulle, S. (2017). Profiling prediction of kinase inhibitors: toward the virtual assay. Journal of medicinal chemistry, 60(1), 474-485.

## Contact
youweifan1028@gmail.com

