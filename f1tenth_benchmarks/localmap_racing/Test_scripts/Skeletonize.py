from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt

# Load an example binary image
image = data.horse()  # Binary image

# Skeletonize the image
skeleton = skeletonize(image)

# Plot the original and skeletonized image
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(skeleton, cmap='gray')
ax[1].set_title('Skeletonized Image')
plt.show()
