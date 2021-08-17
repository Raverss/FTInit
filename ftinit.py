from keras import initializers
from keras.backend import image_data_format, floatx
import Kernels
import math
import random
import numpy as np
np.set_printoptions(precision=2)

class FtInit(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        assert dtype in [float, floatx(), np.float32], "Initializer accepts only float, but got " + dtype
        assert image_data_format() == "channels_last", "expected keras.image_data_format() to be channels_last, but is "+image_data_format()
        assert shape[0] == shape[1], "shape[0] and shape[1] must be equal but are {} and {}".format(shape[0], shape[1])
        assert shape[0] in [3,5,7], "Only kernel shapes 3, 5 and 7 are supported, received {}".format(shape[0])
        params = [
            ('ft0', shape[0], 0, 1, 0),
            ('ft0', shape[0], 0, -1, 0), 
            ('ft1', shape[0], 0, 1, 0), 
            ('ft1', shape[0], 90, 1, 0), 
            ('ft1', shape[0], 0, 1, 45), 
            ('ft1', shape[0], 90, 1, 45), 
            ('ft2c', shape[0], 0, 1, 0), 
            ('ft2c', shape[0], 0, -1, 0)
            ]
        kernels_list = Kernels.get_kernels(params)
        output = np.zeros(shape, dtype=np.float32)
        # for each output filter
        for j in range(shape[3]):
            #init dup to generate at least once
            dup = True
            while(dup):
                # for each input channel
                for i in range(shape[2]):
                    kernel_id = random.randint(0, len(kernels_list)-1)
                    output[..., i,j] = np.copy(kernels_list[kernel_id])
                # normalise
                output[...,j] /= np.max(np.abs(output[...,j]))
                # the .87962566103423978 is std of truncated normal with [-2,2] interval and is taken from scipy see:
                # https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/init_ops_v2.py#L551
                # output[...,j] *= 2*(math.sqrt(2/np.prod(output[...,j].shape[:3]))/.87962566103423978) # normal
                output[...,j] *= math.sqrt(6/np.prod(output[...,j].shape[:3])) # uniform
                # assume that generated filter is not dup
                dup = False
                for k in range(j):
                    # if generated kernel is actually dup, set flag and generate next sequence
                    if (output[...,k] == output[...,j]).all(): dup = True
        return output


class ClassicKernelsInit(initializers.Initializer):
    def __call__(self, shape, dtype=floatx()):
        assert dtype in [float, floatx(), np.float32], "Initializer accepts only float, but got " + dtype
        assert image_data_format() == "channels_last", "expected keras.image_data_format() to be channels_last, but is "+image_data_format()
        assert shape[0] == shape[1], "shape[0] and shape[1] must be equal but are {} and {}".format(shape[0], shape[1])
        assert shape[0] in [3,5,7], "Only kernel shapes 3, 5 and 7 are supported, received {}".format(shape[0])
        params = [
            ('gauss', shape[0], 0, 1, 0),
            ('gauss', shape[0], 0, -1, 0), 
            ('sobel', shape[0], 0, 1, 0), 
            ('sobel', shape[0], 90, 1, 0), 
            ('sobel', shape[0], 0, 1, 45), 
            ('sobel', shape[0], 90, 1, 45),
            ('log', shape[0], 0, 1, 0), 
            ('log', shape[0], 0, -1, 0)
            ]
        kernels_list = Kernels.get_kernels(params)
        output = np.zeros(shape, dtype=np.float32)
        # for each output filter
        for j in range(shape[3]):
            #init dup to generate at least once
            dup = True
            while(dup):
                # for each input channel
                for i in range(shape[2]):
                    kernel_id = random.randint(0, len(kernels_list)-1)
                    output[..., i,j] = np.copy(kernels_list[kernel_id])
                '''
                # we need to standartize before checking for dups as all accepted filters are already standartized
                output[...,j] -= output[...,j].mean()
                output[...,j] /= output[...,j].std()
                # truncated normal should be from [-2*std, 2*std] width std = sqrt(2/fan_in) / .87962566103423978
                # truncated uniform should be from [-sqrt(6/n), +sqrt(6/n)]
                output[...,j] *= 2*math.sqrt(2 / np.prod(output[...,j].shape[:3]))
                '''
                # normalise
                output[...,j] /= np.max(np.abs(output[...,j]))
                # the .87962566103423978 is std of truncated normal with [-2,2] interval and is taken from scipy see:
                # https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/init_ops_v2.py#L551
                # output[...,j] *= 2*(math.sqrt(2/np.prod(output[...,j].shape[:3]))/.87962566103423978) # normal
                output[...,j] *= math.sqrt(6/np.prod(output[...,j].shape[:3])) # uniform
                # assume that generated filter is not dup
                dup = False
                for k in range(j):
                    # if generated kernel is actually dup, set flag and generate next sequence
                    if (output[...,k] == output[...,j]).all(): dup = True
        return output   
