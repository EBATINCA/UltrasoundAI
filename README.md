# UltrasoundAI
3D Slicer extension containing modules to deploy deep learning models for classification and segmentation of ultrasound images.

## Prerequisites
* Install **SlicerOpenCV** extension from the 3D Slicer Extension Manager.
* Install PyTorch in 3D Slicer's Python console: [install PyTorch](https://pytorch.org/). 
For that, open the Python interactor in Slicer and type the following command:</br>
```python
pip_install('torch torchvision torchaudio')
```
* To load the deep learning model used in the module "Breast Lesion Segmentation", install the library [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) in 3D Slicer.
For that, open the Python interactor in Slicer and type the following command:</br>

```python
pip_install('segmentation-models-pytorch')
```

## Information about the Dataset
The models used for classification and segmentation have been trained using [Breast Ultrasound Images Dataset (Dataset BUSI)](https://www.sciencedirect.com/science/article/pii/S2352340919312181)</br>

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863

## Running the training notebook for BreastLesion Segmentation
### Installing Torch + Cuda
You will need a graphics card from Nvidia to properly run Cuda. There are workarounds to make other graphic cards work, but they have to be configured manually.
1. Download and install the Cuda software from Nvidia: https://developer.nvidia.com/cuda-downloads
2. Setup a Python environment. (I've used Python 3.7.13)
3. Install the torch and cuda package: `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html` 

### Other libraries:
`pip install torch-utils
pip install tensorboard
pip install segmentation_models_pytorch
pip install torchsummary`

## Other libraries
### SimpleITK
Needed for working with NIFTI files.
`conda install -c simpleitk simpleitk` (if using anaconda)
or 
`pip install SimpleITK`

## Results
A comparison with two different .pth files, generated in different computers:
![comparison](https://user-images.githubusercontent.com/10054456/170006732-3d9981be-984a-4a6f-8cc2-e7a3976f0d7f.png)
