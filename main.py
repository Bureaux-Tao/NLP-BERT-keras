import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from tensorflow.python.client import device_lib

print(sys.version)
print(sys.executable)

print(K.tensorflow_backend._get_available_gpus())
print(device_lib.list_local_devices())
