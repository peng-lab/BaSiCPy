#!/usr/bin/env python

# # Brain WSI sample
# The notebook is an example of illumination calculation and correction on a sample of images.

# ### Importing the package

# In[1]:


from matplotlib import pyplot as plt
import pybasic


# ### Reading the input images

# In[2]:


wsi_path = "data/WSI_Brain/Uncorrected_tiles/"
WSI_Brain_images = pybasic.tools.load_data(wsi_path, ".tif", verbosity=True)


# ## Run PyBaSiC background calculator

# In[3]:


flatfield, darkfield = pybasic.basic(WSI_Brain_images, darkfield=True)


# ## Show output flatfield and darkfield components

# In[4]:


plt.title("Flatfield")
plt.imshow(flatfield)
plt.colorbar()
plt.show()
plt.title("Darkfield")
plt.imshow(darkfield)
plt.colorbar()
plt.show()


# ### Computing the Timelaps Drift

# In[5]:


baseflour = pybasic.background_timelapse(
    images_list=WSI_Brain_images, flatfield=flatfield, darkfield=darkfield
)


# In[6]:


plt.plot(baseflour)
plt.ylabel("baseline drift")
plt.xlabel("image number")
plt.show()


# ### Applying Background Correction

# In[7]:


WSI_Brain_images_corrected = pybasic.correct_illumination(
    images_list=WSI_Brain_images,
    flatfield=flatfield,
    darkfield=darkfield,
    background_timelapse=baseflour,
)


# ### Comparison Between Corrected and Non-corrected Images
#
# We explore the background correction for the ith image (out of 63 total images).

# In[8]:


i = 24
plt.title("Uncorrected Image")
plt.imshow(WSI_Brain_images[i])
plt.colorbar()
plt.show()
plt.title("Corrected Image")
plt.imshow(WSI_Brain_images_corrected[i])
plt.colorbar()
plt.show()
