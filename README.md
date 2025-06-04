
# Combating Domain Bias: A Learning-then-Generalization Dual Teacher Framework for Unsupervised Domain Adaptation in LiDAR Semantic Segmentation


## Getting Started
```Shell
conda create -n ltgdt python=3.8 -y
conda activate ltgdt

pip install ninja
conda install openblas-devel -c anaconda
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

pip install tensorboard
pip install setuptools==52.0.0
pip install six
pip install pyyaml
pip install easydict
pip install gitpython
pip install wandb
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install tqdm
pip install pandas
pip install scikit-learn
pip install opencv-python
```
pip install other packages if needed.

Our released implementation is tested on
+ Ubuntu 20.04
+ Python 3.8 
+ PyTorch 1.10.1
+ MinkowskiEngine 0.5.4
+ NVIDIA CUDA 11.3
+ 4090 GPU

## Acknowledgement
Thanks for the following works for their awesome codebase.

[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

[SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)

[CoSMix](https://github.com/saltoricristiano/cosmix-uda)

[LaserMix](https://github.com/ldkong1205/LaserMix)


# Code coming soon, stay tuned! 🔥
