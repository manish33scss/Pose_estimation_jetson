
# Pose Estimation Jetson Nano

This a pose estimation module for jetson nano. 


Before begining with pose estimation, few pointers, nvidi provides deep learning 
models, you can follow these steps for install jetson deeplearning models. 

```bash
sudo apt-get update
sudo apt-get install git cmake libpython3-dev python3-numpy
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make -js(nproc)
sudo make install
sudo ldconfig
```
# After installation
RUN
```bash
jetson_test.py
```
#
 This module requires pytorch and 
 torchvision which can either be installed by using the above steps 
which will install deep learning models as well as torch and torchvision(cuda)
 or can following steps for installing only pytorch and torchvision:

 step 0: 
 ```bash
 apt-get install libjpeg-dev zlib1g-dev
 ```
step 1: 
```bash
apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
wget -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl
pip3 install torch-1.9.0-cp36-cp36m-linux_aarch64.whl
```
step 2: 
```bash
sudo apt-get install libjpeg-dev zlib1g-dev libfreetype6-dev
wget https://github.com/pytorch/vision/archive/v0.10.0.tar.gz
tar -xvzf v0.10.0.tar.gz
cd vision-0.10.0
#This takes very long time.
sudo python3 setup.py install
```
step 4:
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python3 setup.py install --plugins
```
step 5:
```bash
pip3 install tqdm cython pycocotools
apt-get install python3-matplotlib
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python3 setup.py install
```

```bash
cd tasks/human_pose
git clone https://github.com/manish33scss/Pose_estimation_jetson.git

```
download weight: [here](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) 
download optimized model: [here](https://drive.google.com/uc?export=download&id=12CzNet267_Pm42Ip7rd3JF3WgwhErGX)
