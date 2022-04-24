import vtk, qt, ctk, slicer
import os
import numpy as np
import time

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import logging
import torch
import cv2

#------------------------------------------------------------------------------
#
# BreastLesionSegmentation
#
#------------------------------------------------------------------------------
class BreastLesionSegmentation(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "BreastLesionSegmentation"
    self.parent.categories = ["Ultrasound AI"]
    self.parent.dependencies = []
    self.parent.contributors = ["David Garcia Mato (Ebatinca), Maria Rosa Rodriguez Luque"]
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
    self.ui.startSegmentationButton.clicked.connect(self.onStartSegmentationButtonClicked)

  #------------------------------------------------------------------------------
  def disconnect(self):
    self.ui.inputSelector.currentNodeChanged.disconnect()
    self.ui.startSegmentationButton.clicked.disconnect()

  #------------------------------------------------------------------------------
  def updateGUIFromMRML(self, caller=None, event=None):
    """
    Set selections and other settings on the GUI based on the parameter node.

    Calls the updateGUIFromMRML function of all tabs so that they can take care of their own GUI.
    """    
    # Activate buttons
    self.ui.startSegmentationButton.enabled = (self.ui.inputSelector.currentNode() != None)

  #------------------------------------------------------------------------------
  def onInputSelectorChanged(self):
    
    # Update GUI
    self.updateGUIFromMRML()

  #------------------------------------------------------------------------------
  def onStartSegmentationButtonClicked(self):
    # Get input volume
    inputVolume = self.ui.inputSelector.currentNode()

    # Get image data
    self.logic.getImageData(inputVolume)

    # Segmentation
    self.logic.startSegmentation()         

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

  #------------------------------------------------------------------------------
  def getImageData(self, volumeNode):

    # Get numpy array from volume node
    self.imageArray = slicer.util.arrayFromVolume(volumeNode)[0]

  #------------------------------------------------------------------------------
  def startSegmentation(self):
    """
    Image segmentation.
    """
    print('Starting segmentation...')

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
