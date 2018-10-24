
from model import Model, ModelTorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import settings


from unbin import getCooccurrenceMatrix
from parse_concept_net import getRelationMatrices
print(len(getRelationMatrices(settings.vocab_size)))
# print(len(getCooccurrenceMatrix(settings.vocab_size)))