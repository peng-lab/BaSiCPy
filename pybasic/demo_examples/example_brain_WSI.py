import numpy as np
import os
from matplotlib import pyplot as plt
import pybasic

path_to_demo_examples = 'path/to/demoexamples/'
path_to_demo_examples = '/Users/lorenz.lamm/PhD_projects/BaSiC/Demoexamples'
wsi_path = os.path.join(path_to_demo_examples, 'WSI_Brain/Uncorrected_tiles')

D = pybasic.tools.load_data(wsi_path, '.tif', working_size=(128, 128), interpolation_method='bilinear')

flatfield, darkfield = pybasic.basic(D, segmentation=None, darkfield=True) # Run BaSiC to estimate both flatfield and darkfield


# Show flatfield and darkfield components
plt.imshow(flatfield)
plt.colorbar()
plt.show()
plt.imshow(darkfield)
plt.colorbar()
plt.show()

# Correct image stack
D_corrected = np.transpose(D, (2,0,1))
D_corrected = np.array(D_corrected, dtype=np.float64)
D_corrected -= darkfield
D_corrected /= flatfield
D_corrected = np.transpose(D_corrected, (1,2,0))

# show comparison between corrected vs. uncorrected image
plt.subplot(1,2,1)
plt.imshow(D[:,:,30])
plt.title('Uncorrected')
plt.subplot(1,2,2)
plt.imshow(D_corrected[:,:,30])
plt.title('Corrected')
plt.show()

