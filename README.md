# Spatio-Temporal Transformer 
STformer: Spatio-Temporal Transformer for Multivariate Time Series Anomaly Detection

Unsupervised detection of anomaly in time series is a challenging problem, which requires the model to learn informative representation and derive a distinguishable criterion. In this paper, we propose the Spatio-Temporal Transformer in these three folds:

- A **multi head patches attention** mechanism for multivariate time series across different dimensions and temporal patches.
- This paper proposes **Spatio-Temporal Attention**, which aims to capture the spatial dependency of temporal patches and adjacent space.
- **STformer** achieves the **state-of-the-art** on a representative dataset for anomaly detection with **19.69\%** F1 score improvement over the baseline transformer.
<p align="center">
<img src=".\pics\key.png" height = "350" alt="" align=center />
</p>

## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0. 
2. Download data. You can download the dataset used in the experiment and all the code on this webpage from the [Dropbox link](https://www.dropbox.com/sh/jtpmdoxyg2ybolh/AABL9oYTWreFxFBAgqOGxafJa?dl=0). Unzip file `./dataset/data.zip` to the same path `./dataset`, you can get the four kinds of data. **All the datasets are well pre-processed**. 
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the file `./Executable.sh`.This file contains the training and testing code for the four datasets. You can reproduce the experiment results as follows:
```bash
bash ./Executable.sh
```

Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. If you have questions about this, please see this [issue](https://github.com/thuml/Anomaly-Transformer/issues/14) or email us.

## Main Result

We compare our model with 7 baselines, including Transformer, Anomalyformer, etc. **Generally,  Spatio-Temporal Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\results.png" height = "450" alt="" align=center />
</p>


## Contact
If you have any question, please contact zhengyulee321@163.com.
