import os
import numpy as np
from matplotlib import pyplot as plt
import pybasic


path_to_demo_examples = 'path/to/demoexamples/'
path_to_demo_examples = 'C:/Users/Loren/Documents/Basic_Examples/Demoexamples/'
timelapse_path = os.path.join(path_to_demo_examples, 'timelapse_nanog/Uncorrected')
D = pybasic.tools.load_data(timelapse_path, '.tif', working_size=(128, 128), interpolation_method='bilinear')
flatfield, darkfield = pybasic.basic(D, segmentation=None, darkfield=True, lambda_s=2.0, lambda_darkfield=2.0)

images_list = []
for i in range(D.shape[2]):
    images_list.append(D[:,:,i])
baseflour = pybasic.background_timelapse(images_list, flatfield, darkfield)

plt.figure()
plt.imshow(flatfield)
plt.colorbar()

plt.figure()
plt.imshow(darkfield)
plt.colorbar()

plt.figure()
plt.plot(baseflour)
plt.show()

# Image correction
D_corr = np.zeros(D.shape)
for i in range(D_corr.shape[2]):
    D_corr[:,:,i] = (D[:,:,i] - darkfield) / flatfield - baseflour[i]

plt.figure()
plt.subplot(1,2,1)
plt.imshow(D[:,:,0])
plt.title('Uncorrected')
plt.subplot(1,2,2)
plt.imshow(D_corr[:,:,0])
plt.title('Corrected')
plt.show()
