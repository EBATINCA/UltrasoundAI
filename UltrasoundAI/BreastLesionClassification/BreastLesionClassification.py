import vtk, qt, ctk, slicer
import os
import numpy as np
import time

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import logging
import PIL
from PIL import Image

# Load PyTorch library
try:
  import torch
  import torchvision
  from torch.autograd import Variable
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # define current device
except:
  logging.error('PyTorch is not installed. Please, install PyTorch...')

# Load OpenCV
try:
  import cv2
except:
  logging.error('OpenCV is not installed. Please, install OpenCV...')


#------------------------------------------------------------------------------
#
# BreastLesionClassification
#
#------------------------------------------------------------------------------
class BreastLesionClassification(ScriptedLoadableModule):
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Breast Lesion Classification"
    self.parent.categories = ["Ultrasound AI"]
    self.parent.dependencies = []
    self.parent.contributors = ["David Garcia Mato (Ebatinca), Daria Dona Falcon (ULPGC)"]
    self.parent.helpText = """ Module to classify breast lesions in US images using deep learning. """
    self.parent.acknowledgementText = """EBATINCA, S.L."""


#------------------------------------------------------------------------------
#
# BreastLesionClassificationWidget
#
#------------------------------------------------------------------------------
class BreastLesionClassificationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

  def __init__(self, parent):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation

    # Create logic class
    self.logic = BreastLesionClassificationLogic(self)

  #------------------------------------------------------------------------------
  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Set up UI
    self.setupUi()

    # Setup connections
    self.setupConnections()

    # The parameter node had defaults at creation, propagate them to the GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def cleanup(self):
    self.disconnect()

  #------------------------------------------------------------------------------
  def enter(self):
    """
    Runs whenever the module is reopened
    """

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def exit(self):
    """
    Runs when exiting the module.
    """
    pass

  #------------------------------------------------------------------------------
  def setupUi(self):    
    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/BreastLesionClassification.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Customize widgets
    self.ui.PathLineEdit.currentPath = self.resourcePath('Model/classification_model.pth')

  #------------------------------------------------------------------------------
  def setupConnections(self):    
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.ui.loadModelButton.connect('clicked(bool)', self.onloadModelButton)

  #------------------------------------------------------------------------------
  def disconnect(self):
    self.ui.applyButton.clicked.disconnect()
    self.ui.loadModelButton.clicked.disconnect()

  #------------------------------------------------------------------------------
  def updateGUIFromMRML(self, caller=None, event=None):
    """
    Set selections and other settings on the GUI based on the parameter node.

    Calls the updateGUIFromMRML function of all tabs so that they can take care of their own GUI.
    """    
    # Activate buttons
    

    # Display selected volume in red slice view
    inputVolume = self.ui.inputSelector.currentNode()
    if inputVolume:
      self.logic.displayVolumeInSliceView(inputVolume)

  #------------------------------------------------------------------------------
  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    
    # Get input volume
    inputVolume = self.ui.inputSelector.currentNode()
    # Get image data
    self.logic.getImageData(inputVolume)

    # Classification
    [valBen, valMal, valNor, mostProbableClass] = self.logic.startClassification()

    # Display probabilities in UI
    self.ui.benignValue.text= str(valBen)
    self.ui.malignValue.text = str(valMal)
    self.ui.normalValue.text = str(valNor)

    # Display most probable class in UI
    self.ui.mostProbableClassLabel.text = mostProbableClass

  #------------------------------------------------------------------------------
  def onloadModelButton(self):
  #Acquire path from the line in UI
   modelFilePath = self.ui.PathLineEdit.currentPath
   #Load model using the function in the logic section
   self.logic.loadModel(modelFilePath)
   if self.logic.model is not None:
    print("Model loaded")

#------------------------------------------------------------------------------
#
# BreastLesionClassificationLogic
#
#------------------------------------------------------------------------------
class BreastLesionClassificationLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
  
  def __init__(self, widgetInstance, parent=None):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self, parent)
    VTKObservationMixin.__init__(self)

    # Red slice
    self.redSliceLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()

    # Classification model
    self.model = None

  #------------------------------------------------------------------------------
  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
  
  #------------------------------------------------------------------------------
  def getImageData(self, volumeNode):

    # Get numpy array from volume node
    self.imageArray = slicer.util.arrayFromVolume(volumeNode)[0]

    # Get image dimensions
    self.numRows = self.imageArray.shape[0]
    self.numCols = self.imageArray.shape[1]
    print('Image dimensions: [%s, %s]' % (str(self.numRows), str(self.numCols)))

    # Get image statistics
    maxValue = np.max(self.imageArray)
    minValue = np.min(self.imageArray)
    avgValue = np.mean(self.imageArray)
    print('Image maximum value = ', maxValue)
    print('Image minimum value = ', minValue)
    print('Image average value = ', avgValue)
  
  #------------------------------------------------------------------------------
  def displayVolumeInSliceView(self, volumeNode):
    # Display input volume node in red slice background
    self.redSliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())
    self.redSliceLogic.FitSliceToAll()
  
  #------------------------------------------------------------------------------
  def loadModel(self, modelFilePath):
    """
    Tries to load PyTorch model for segmentation
    :param modelFilePath: path where the model file is saved
    :return: True on success, False on error
    """
    print('Loading model...')

    #Use torch function to load model from given path
    try:
      self.model = torch.load(modelFilePath)
    except:
      self.model = None
      return False
    
    return True
  
  #------------------------------------------------------------------------------
  def startClassification(self):
    """
    Image classification.
    """
    #Definition of transformations applied to image (resizing, transformation into tensor and normalizing values). These are necessary to go through the model
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.CenterCrop(299),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #To allow segmentation in phantom images (1 channel)
    #If you comment this line the result is the same in Dataset BUSI (3 Channels)
    img=cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2RGB)

    #Transform the image from numerical array to PIL Image format
    image = Image.fromarray(img)

    #Previously defined transforms are applied, plus image is unsqueezed to have enough dimensions to go through model
    image_trans = data_transforms(image).float()
    image_var = Variable(image_trans, requires_grad=True)
    image_clas = image_var.unsqueeze(0)

    print('Starting classification...')
    if self.model is None:
     print("No model loaded")

    #Image is taken through model, which gives a tensor with one value per class as output
    tens = self.model(image_clas)

    #Values are turned into percentage and rounded to 2 decimals
    percentage = torch.nn.functional.softmax(tens, dim=1)[0] * 100
    valMal = percentage.data.cpu().numpy()[1]
    valBen = percentage.data.cpu().numpy()[0]
    valNor = percentage.data.cpu().numpy()[2]
    valMal = round(valMal,2)
    valBen = round(valBen,2)
    valNor = round(valNor,2)

    # Get most probable class
    maxValue = max(valMal, valBen, valNor)
    if maxValue == valMal:
      mostProbableClass = 'Malignant'
    elif maxValue == valBen:
      mostProbableClass = 'Benign'
    elif maxValue == valNor:
      mostProbableClass = 'Normal'
    else:
      mostProbableClass = 'None'

    #Output is the percentage corresponding to each class plus the most probable class expressed as a string
    return valBen, valMal, valNor, mostProbableClass


#------------------------------------------------------------------------------
#
# BreastLesionClassificationTest
#
#------------------------------------------------------------------------------
class BreastLesionClassificationTest(ScriptedLoadableModuleTest):
  """This is the test case for your scripted module.
  """
  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    ScriptedLoadableModuleTest.runTest(self)
