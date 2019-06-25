import numpy as np
import chainer
import argparse
import os
from chainercv.datasets import DirectoryParsingLabelDataset

import chainer                                                                                        
from chainer import training                                                                        
from chainer.training import extensions     
from chainercv.datasets import DirectoryParsingLabelDataset 


def tuple2array(url):

    t = DirectoryParsingLabelDataset(url) 


    new = [np.arange(3 * 32 * 32,dtype = 'float32').reshape(3, 32, 32)] * len(t)

    for i in range(len(t)):
        new[i] = t[i][0]
    
    return new