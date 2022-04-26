# UltrasoundAI
3D Slicer extension containing modules to deploy deep learning models for classification and segmentation of ultrasound images.

## Prerequisites
* From 3D Slicer Extension Manager: **SlicerOpenCV**<br> (View-> Extension Manager-> SlicerOpenCV-> Install)
* Check that you have installed PyTorch in Slicer's Python: [install PyTorch](https://pytorch.org/).
* To load the deep learning model used in the module "Breast Lesion Segmentation", install the library **segmentation-models-pytorch** in 3D Slicer.
For that, open the Python interactor in Slicer and type the following command:</br>
```python
pip_install('segmentation-models-pytorch')
```
