# CGGLNet  EIGNet

CGGLNet: Semantic Segmentation Network for  Remote Sensing Images Based on Category-Guided  Global-Local Feature Interaction（IEEE Transactions on Geoscience and Remote Sensing，DOI: 10.1109/TGRS.2024.3379398）

EIGNet：Edge Guidance Network for Semantic Segmentation of High Resolution Remote Sensing Images（IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing，DOI: 10.1109/JSTARS.2023.3316307）


Our training and testing framework follows UNetFormer (https://github.com/WangLibo1995/GeoSeg), which is a well-established framework, thanks to the author LiBo Wang for sharing it.


Simply add config, losses and models to its framework and it is ready to run.The execution command is in configs.txt.


The relevant code is being organized and will be posted as soon as possible.

If you don't want to retrain, we will also upload the data from the dataset in the following, as well as the weights of the relevant networks for direct testing.

After finishing, we will release the Baidu cloud URL.

Link https://pan.baidu.com/s/1bY_oqsmkLe5PpvDhzM1HQg?pwd=l1qt  
xtract code: l1qt

If you find this project useful in your research, please consider citing：

EIGNet：
@ARTICLE{10254241,
  author={Ni, Yue and Liu, Jiahang and Cui, Jian and Yang, Yuze and Wang, Xiaozhen},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Edge Guidance Network for Semantic Segmentation of High-Resolution Remote Sensing Images}, 
  year={2023},
  volume={16},
  number={},
  pages={9382-9395},
  keywords={Feature extraction;Semantics;Semantic segmentation;Image edge detection;Remote sensing;Convolution;Data mining;Edge information;orientation astrous convolution;remote sensing;semantic segmentation},
  doi={10.1109/JSTARS.2023.3316307}}
  
CGGLNet：
@ARTICLE{10475326,
  author={Ni, Yue and Liu, Jiahang and Chi, Weijian and Wang, Xiaozhen and Li, Deren},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CGGLNet: Semantic Segmentation Network for Remote Sensing Images Based on Category-Guided Global–Local Feature Interaction}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Transformers;Context modeling;Remote sensing;Semantic segmentation;Data mining;Adaptation models;Category-guided;global contextual information;remote sensing;semantic segmentation;transformer},
  doi={10.1109/TGRS.2024.3379398}}

