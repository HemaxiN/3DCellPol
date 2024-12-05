import numpy as np
import os
import matplotlib.pyplot as plt

prediction_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results' #prediction folder defined in predict_main.py
images_dir = r'/home/jovyan/discotreino/3DCellPol_Github/images' # directory containing the original  .tif images, image shape = (x,y,z,channels)
dimx = 0.666 #resolution across the x direction in µm
dimy = 0.666 #resolution across the y direction in µm
dimz = 0.270 #resolution across the z direction in µm
plane_ = 'xy' # choose between 'xy', 'yz' or 'zx'
image_names = [f for f in os.listdir(images_dir) if f.endswith('.tif')] ## image names


vectors = []
def distance_(p1,p2):
    dist_ = np.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2) + ((p1[2] - p2[2])**2))
    return dist_

for iname in image_names:
    nuclei_centroids = np.load(os.path.join(prediction_dir, iname.replace('.tif','_nuclei_centroids_nms.npy'))) 
    golgi_centroids = np.load(os.path.join(prediction_dir, iname.replace('.tif','_golgi_centroids_nms.npy')))
    for N, G in zip(nuclei_centroids, golgi_centroids):
        XN = int(N[0])*dimx
        YN = int(N[1])*dimy
        ZN = int(N[2])*dimz
        XG = int(G[0])*dimx
        YG = int(G[1])*dimy
        ZG = int(G[2])*dimz
        vectors.append((XG-XN, YG-YN, ZG-ZN))
        
vectors = np.asarray(vectors)

if plane_ == 'xy':
    # Calculate the azimuth angle (angle in the xy-plane)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # y, x components
elif plane_ == 'yz':
    angles = np.arctan2(vectors[:, 1], vectors[:, 2])  # y, z components
elif plane_ == 'zx':
    angles = np.arctan2(vectors[:, 2], vectors[:, 0])  # z, x components
else:
    print('Select a plane from {xy, yz, zx}')

# Normalize to range [0, 2*pi]
angles = np.mod(angles, 2*np.pi)  
# Set the number of angular bins
num_bins = 36
# Create the histogram for azimuth angles and normalize to probabilities
counts, bin_edges = np.histogram(angles, bins=num_bins)
# Normalize to probabilities
counts_normalized = counts / np.sum(counts)  

# Create polar plot with normalized counts
plt.figure()
ax = plt.subplot(111, polar=True)
bars = ax.bar((bin_edges[:-1] + bin_edges[1:]) / 2, counts_normalized, width=np.diff(bin_edges), align='center')
ax.set_ylim(0, max(counts_normalized)+0.05)
radial_ticks = np.arange(0, max(counts_normalized)+0.1, 0.05)  # Adjust step as needed
ax.set_yticks(radial_ticks)
# Increase font size of radial ticks
ax.tick_params(axis='y', labelsize=20)  # Adjust 14 to desired font size
# Increase font size of angular ticks
ax.tick_params(axis='x', labelsize=20)  # Adjust 14 to desired font size
plt.savefig(os.path.join(prediction_dir, 'Polar Histogram {}.jpg'.format(plane_)), dpi=100)