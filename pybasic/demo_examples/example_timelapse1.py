import os
from matplotlib import pyplot as plt
import pybasic

path_to_demo_examples = 'path/to/demoexamples/'
path_to_demo_examples = 'C:/Users/Loren/Documents/Basic_Examples/Demoexamples/'
timelapse_path = os.path.join(path_to_demo_examples, 'timelapse_brightfield/Uncorrected')
D = pybasic.tools.load_data(timelapse_path, '.png', working_size=(128, 128), interpolation_method='bilinear')
flatfield = pybasic.basic(D, segmentation=None)

images_list = []
for i in range(D.shape[2]):
    images_list.append(D[:,:,i])
baseflour = pybasic.background_timelapse(images_list, flatfield)

plt.figure()
plt.imshow(flatfield)
plt.figure()
plt.plot(baseflour)
plt.show()
