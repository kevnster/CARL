# IMPORTS
import copy, keyboard, math, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from ai2thor.controller import Controller
from PIL import Image, ImageDraw

# matplotlib.use() // REPLACE WITH BACKEND

class 2D(object):
    def __init__(self, frame):