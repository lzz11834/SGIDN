 <h1 align="center"> Project-SGIDN </h1>
This repository provides a tensorflow implementation of the *SGIDN* method presented in our
RSE 2020 paper  ”SGIDN”.

<p align="center">
  <a href="https://github.com/TulioOParreiras/ProjectTemplate/issues">
    </a>
    <img src="https://github.com/lzz11834/SGIDN/blob/master/imgs/Chapter.3-Fig.3.png" />
	<br>
	<br>
</p

Abstract

From the EO-1 Hyperion imaging spectrometer to the newly launched Chinese satellite hyperspectral imagers, stripe noise is a ubiquitous phenomenon that seriously degrades the data quality and usability. Although previous efforts have achieved inspiring results, hyperspectral image (HSI) destriping remains a challenging task, as the stripe degradations are sometimes more complicated than the predefined assumptions, i.e., the preselected reference, filter, or handcrafted priors. With the rapid advances in deep learning technologies, convolutional neural networks (CNNs) provide a new potential to learn essential priors in an automatic manner. However, the training of CNNs is highly reliant on a large high-quality standard dataset, which is difficult to acquire for hyperspectral spaceborne sensors. In this paper, an innovative approach termed the satellite-ground integrated destriping network (SGIDN) is proposed for HSIs. Rather than using self-training, a satellite-ground integrated strategy is proposed, for the first time, to mitigate the data dependency, so that a large set of striped-clean pairs is generated from the ground-based HSIs. Considering the varied stripes among different bands, a unique CNN architecture design, including the combination of 3D convolution and 2D convolution, residual learning, and supplementary gradient channels, is integrated to capture the intrinsic spectral-spatial features in the HSIs and the unidirectional property of stripe noise. Compared with the traditional methods, SGIDN can be flexibly extended to specific HSI destriping tasks, e.g., coexisting horizontal and vertical stripes, and generalizes well to different hyperspectral satellite sensors. Given the same study area (Shanghai, China), three HSIs acquired by the EO-1 Hyperion imaging spectrometer, the Chinese HJ-1A HSI sensor, and the wide-range hyperspectral imager onboard the Chinese SPARK spectral micro-nano satellite are adopted to assess the proposed SGIDN model. Both simulated and real-data experiments confirm that SGIDN can consistently outperform the benchmark methods, with a higher degree of efficiency. Moreover, the land-cover mapping results further demonstrate the necessity of destriping and the suitability of the destriped results for use in further applications.

## 👨🏻‍💻 Citation and Contact

If you use our work, please also cite the paper:

```
@ARTICLE{9007624,  
 author={Y. {Zhong} and W. {Li} and X. {Wang} and S. {Jin} and L. {Zhang}},
 journal={Remote Sensing of Environment},
 title={Satellite-ground integrated destriping network: A new perspective for EO-1 Hyperion and Chinese hyperspectral satellite datasets},
 year={2020},
 volume={237},
 number={},
 pages={111416},
 }
```

If you would like to get in touch, please contact [liwenqing_rs@whu.edu.cn](mailto:luozhaozhi2016@whu.edu.cn).

## 📋 Requirements

This code is written in `Python 3.6` and requires the packages listed in `requirements.txt`.

To run the code, we recommend setting up a virtual environment, e.g. using `conda`:

## ✨ Features
* The image flowers from the ground-based hyperspectral dataset CAVE for training
* Washington DC Mal for test


## 📲 Installation
### `conda`

```
cd <path>
conda create -n your_env_name python=3.6
conda activate your_env_name
conda install --yes --file requirements.txt
```

## 🚀 Running experiments

We have implemented the DC dataset.

### DC example

```
cd <path>

# activate your virtual environment
conda activate your_env_name

# run experiment
python generate_patches.py
python train.py
python test.py
```
Note: the depth of input in this model is 10.

## ❤️ Examples
The results of these examples are put in './data/result/examples'


## 👮🏻‍♂️ License
The copyright belongs to Intelligent Data Extraction, Analysis and Applications of Remote Sensing(RSIDEA) academic research group,State Key Laboratory of Information Engineering in Surveying, Mapping, and Remote Sensing (LIESMARS),Wuhan University.
This program is for academic use only. For commercial use, please contact Professor.Zhong(zhongyanfei@whu.edu.cn).
The homepage of RSIDEA is: http://rsidea.whu.edu.cn/index.html
