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

