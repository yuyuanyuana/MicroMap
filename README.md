# 🧬 MicroMap: High-Resolution Spatial Transcriptomics from H&E-stained histological images

## 📖 Overview
MicroMap is a deep generative framework for predicting high-resolution spatial transcriptomic (ST) profiles directly from H&E-stained histological images. By leveraging an spot level expression guided deep generative model, MicroMap enables accurate in-sample resolution enhancement and direct out-of-sample high-resolution gene expression prediction from H&E image only. Overall, MicroMap improves effective spatial resolution and prediction, offering a practical, affordable solution for spatial molecular analysis.

## 🚀 Installation

### System Requirements
- **Operating System:** Ubuntu 22.04
- **Python:** 3.9.21 (tested)
- **Dependencies:** See `requirements.txt` for full list. Key packages include PyTorch, NumPy, pandas, scikit-learn, matplotlib
- **Hardware:** GPU recommended for training (tested on NVIDIA A800 80GB PCIe, Driver 550.144.03, CUDA 12.4)

### Install from GitHub
First, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, install MicroMap directly from GitHub (this should only take a few seconds):

```bash
https://github.com/yuyuanyuana/MicroMap/releases/download/v1.0.0/MicroMap-1.0.0-py3-none-any.whl
```

## 📘 Tutorial
For detailed usage examples, please refer to the in-sample tutorial notebook and the out-of-sample tutorial notebook , 
- [`in_sample`](tutorial_in_sample.ipynb)
- [`out-of-sample`](tutorial_out_of_sample.ipynb)
which demonstrate the complete workflow, including data download and loading, model training and prediction, and quantitative evaluation.

## 📊 Benchmarking 
MicroMap is benchmarked against four state-of-the-art high-resolution prediction methods: 
- iStar: https://github.com/daviddaiweizhang/istar
- XFuse: https://github.com/ludvb/xfuse
- TESLA: https://github.com/jianhuupenn/TESLA
- MISO: https://github.com/owkin/miso_code
For iStar and XFuse, we use the default training and inference settings provided by the official implementations.
To enable dense whole-slide prediction rather than region-limited spot-level prediction, we make minor modifications to the inference pipelines of TESLA and MISO while keeping their training configurations unchanged.
The modified evaluation scripts are available at:
https://github.com/ToryDeng/MicroMap-analysis


## 📄 License
MicroMap 1.0.0 is released under the MIT License. See the LICENSE file for details.


## 📑 Citation
coming soon.

