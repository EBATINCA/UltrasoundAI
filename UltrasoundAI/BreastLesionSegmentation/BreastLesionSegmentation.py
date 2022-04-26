import vtk, qt, ctk, slicer
import os
import numpy as np
import time

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import logging

# Load PyTorch library
try:
  import torch
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
# BreastLesionSegmentation
#
#------------------------------------------------------------------------------
class BreastLesionSegmentation(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Breast Lesion Segmentation"
    self.parent.categories = ["Ultrasound AI"]
    self.parent.dependencies = []
    self.parent.contributors = ["David Garcia Mato (Ebatinca), Maria Rosa Rodriguez Luque (ULPGC)"]
    self.parent.helpText = """ Module to segment lesions in US images using deep learning. """
    self.parent.acknowledgementText = """EBATINCA, S.L."""

#------------------------------------------------------------------------------
#
# BreastLesionSegmentationWidget
#
#------------------------------------------------------------------------------
class BreastLesionSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  #------------------------------------------------------------------------------
  def __init__(self, parent):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    
    # Create logic class
    self.logic = BreastLesionSegmentationLogic(self)

    slicer.BreastLesionSegmentationWidget = self # ONLY FOR DEVELOPMENT
    
    self.exist_mask=False # True when "Start segmentation button" is clicked and executed successfully
    self.is_model_loaded=False # True when the model has been loaded
    self.new_model_path=None # Save the path of the model file selected by the user


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
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/BreastLesionSegmentation.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

  #------------------------------------------------------------------------------
  def setupConnections(self):    
    self.ui.inputSelector.currentNodeChanged.connect(self.onInputSelectorChanged)
    self.ui.loadModelButton.clicked.connect(self.loadModelButtonClicked)
    self.ui.startSegmentationButton.clicked.connect(self.onStartSegmentationButtonClicked)
    self.ui.saveMaskButton.clicked.connect(self.onSaveMaskButtonClicked)

  #------------------------------------------------------------------------------
  def disconnect(self):
    self.ui.inputSelector.currentNodeChanged.disconnect()
    self.ui.loadModelButton.clicked.disconnect()
    self.ui.startSegmentationButton.clicked.disconnect()
    self.ui.saveMaskButton.clicked.disconnect()

  #------------------------------------------------------------------------------
  def updateGUIFromMRML(self, caller=None, event=None):
    """
    Set selections and other settings on the GUI based on the parameter node.

    Calls the updateGUIFromMRML function of all tabs so that they can take care of their own GUI.
    """    
    # Activate buttons
    self.ui.loadModelButton.enabled = True
    self.ui.startSegmentationButton.enabled = (self.ui.inputSelector.currentNode() != None and self.is_model_loaded==True)
    self.ui.saveMaskButton.enabled = (self.ui.startSegmentationButton.enabled and self.exist_mask==True)

  #------------------------------------------------------------------------------
  def onInputSelectorChanged(self):

    # Update GUI
    self.updateGUIFromMRML()
 
  #------------------------------------------------------------------------------
  def loadModelButtonClicked(self):
    
    #Read the path of the model file
    self.new_model_path=self.ui.modelPathEdit.currentPath 

    #If there is not a path selected, download the default model in the Resources directory
    if self.new_model_path!='':
      modelFilePath=self.new_model_path
    else:
      modelFilePath=self.resourcePath('Model/segmentation_model.pth')

    # Load model
    self.is_model_loaded = self.logic.loadModel(modelFilePath)

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def onStartSegmentationButtonClicked(self):
    # Get input volume
    inputVolume = self.ui.inputSelector.currentNode()

    # Get image data
    self.logic.getImageData(inputVolume)

    # Prepare data
    self.logic.prepareData()

    # Segmentation
    self.exist_mask=self.logic.startSegmentation(inputVolume)

    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def onSaveMaskButtonClicked(self):

    nodeName=self.ui.inputSelector.currentNode().GetName()
    self.logic.saveMask(self.resourcePath('Data/Predicted_mask/{name}.png'.format(name=nodeName)))

#------------------------------------------------------------------------------
#
# BreastLesionSegmentationLogic
#
#------------------------------------------------------------------------------
class BreastLesionSegmentationLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
  
  #------------------------------------------------------------------------------
  def __init__(self, widgetInstance, parent=None):
    ScriptedLoadableModuleLogic.__init__(self, parent)
    VTKObservationMixin.__init__(self)

    # Image array
    self.imageArray = None
    self.model = None
    self.maskVolume_created=False

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
  
  def prepareData(self):
    """
    Prepare image to be process by the PyTorch Model 
    """
    print('Preparing data...')

    #To allow segmentation in phantom images (1 channel)
    #If you comment this line the result is the same in Dataset BUSI (3 Channels)
    img=cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2RGB) 
    
    #Resize
    img=cv2.resize(img, (256, 256))
    
    #Normalize
    img=img.astype(np.float32)
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    img = np.clip(img, -1.0, 1.0)
    img = (img + 1.0) / 2.0

    #Transform numpy array to tensor
    self.img_prepared = np.transpose(img, (2, 0, 1))

    print('Data prepared!')
  #------------------------------------------------------------------------------

  def loadModel(self, modelFilePath):
    """
    Tries to load PyTorch model for segmentation
    :param modelFilePath: path where the model file is saved
    :return: True on success, False on error
    """
    print('Loading model...')
    try:
      print('Model directory:',modelFilePath)
      self.model = torch.load(modelFilePath, map_location = torch.device(DEVICE))
    except:
      self.model = None
      logging.error("Failed to load model")
      return False
    print('Model loaded!') 
    return True
  
  #------------------------------------------------------------------------------
  def startSegmentation(self,volumeNode):
    """
    Image segmentation
    :return: True on success, False on error
    """
    print('Starting segmentation...the process could take a few seconds')

    #Predict the mask
    try:
      input_image = torch.from_numpy(self.img_prepared).to(DEVICE).unsqueeze(0)
      self.pr_mask = self.model.predict(input_image)
      self.pr_mask = self.pr_mask.squeeze().cpu().numpy().round()
      self.pr_mask = self.pr_mask.astype(np.uint8)*255
    except:
      logging.error("Can not segment the image!")
      return False

    #Resize mask
    self.pr_mask=cv2.resize(self.pr_mask, (self.numCols, self.numRows))
    
    #Check mask and ultrasound image shape
    print('Mask dimensions:',self.pr_mask.shape)
    print('Image dimensions:',self.imageArray.shape)

    #Create the segmentation volume 
    if self.maskVolume_created==False:
       self.getMaskVolume(volumeNode)

    #Update "Segmentation image" volume content
    segmentation=np.expand_dims(self.pr_mask,0)
    print('Segmentation dimensions:',segmentation.shape)

    slicer.util.updateVolumeFromArray(self.outputVolumeNode, segmentation)


    print('Segmentation finished!')
    return True

  #------------------------------------------------------------------------------
  def saveMask(self,maskPath):
    """
    Save predicted segmentation.
    """
    print('Saving mask...')

    try:
      cv2.imwrite(maskPath,self.pr_mask)
    except:
      logging.error("Failed to save mask")
      return
    
    print('Mask saved correctly!')

  #------------------------------------------------------------------------------
  def getMaskVolume(self,volumeNode):
    """
    Save predicted segmentation.
    """
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    inputVolumeItemID = shNode.GetItemByDataNode(volumeNode)
    outputVolumeItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, inputVolumeItemID)
    self.outputVolumeNode = shNode.GetItemDataNode(outputVolumeItemID)
    self.outputVolumeNode.SetName('Segmented image')
    self.maskVolume_created=True
    
#------------------------------------------------------------------------------

    

#------------------------------------------------------------------------------
#
# BreastLesionSegmentationTest
#
#------------------------------------------------------------------------------
class BreastLesionSegmentationTest(ScriptedLoadableModuleTest):
  """This is the test case for your scripted module.
  """
  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    ScriptedLoadableModuleTest.runTest(self)
