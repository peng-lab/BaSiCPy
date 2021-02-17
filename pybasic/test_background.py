from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib import pyplot as plt
import pybasic
import cv2




def get_blank_parameter_dict():
    parameter_dict = {
        'basic_lambda': None,  # Lambda value for smoothness of flatfield
        'estimation_mode': None,  # ???
        'max_iteration': None,  # maximum number of iterations per reweighting step
        'optimization_tol': None,  # tolerance threshold for stopping Criterion
        'darkfield': False,  # Boolean, whether or not darkfield component should be estimated
        'lambda_darkfield': None,  # smoothness parameter for darkfield component
        'initial_flatfield': None,  # Initialization for flatfield component (No optimization for flatfield is done)
        'segmentation': None  # segmentation masks for all images, should be the same shape as input image sequence
    }
    return parameter_dict


path_to_demo_examples = 'path/to/demoexamples/'
path_to_demo_examples = 'C:/Users/Loren/Documents/Basic_Examples/Demoexamples/'
timelapse_path = os.path.join(path_to_demo_examples, 'Timelapse_brightfield/Uncorrected_resized')

D = pybasic.tools.load_data(timelapse_path, '.png', working_size=(128, 128), interpolation_method='bilinear')
parameter_dict = get_blank_parameter_dict()


flatfield = plt.imread('C:/Users/Loren/Documents/pybasic/test_flat.png', 'bw')
flatfield = np.load('C:/Users/Loren/Documents/pybasic/test_flat.arr.npy')
flatfield = cv2.imread('C:/Users/Loren/Documents/Basic_Examples/Demoexamples/Timelapse_brightfield/Uncorrected_resized/test_flat.tif', cv2.IMREAD_GRAYSCALE)
flatfield = flatfield / 255
# darkfield = np.load('C:/Users/Loren/Documents/pybasic/test_dark.arr.npy')
# flatfield = BaSiC(D, parameter_dict) # Run BaSiC to estimate flatfield
images_list = []
for i in range(D.shape[2]):
    images_list.append(D[:,:,i])
baseflour = pybasic.background_timelapse(images_list, flatfield)
print(baseflour.shape)
print(baseflour)
plt.plot(baseflour)
plt.show()
