from engine import DoubleStackEngine

import time


# Start timer
start_time = time.time()

# Initialize engine
dse = DoubleStackEngine()

# Set required information
dse.set('dry', 'brine', 'filtered', 'output')

# Read stacks
dse.read()

# Use to tune image processing parameters and assure correct contour detection
# dse.tune(image_blur_ksize=(25, 25),
#           image_threshold=80,
#           image_blur_sigma=(0, 0),
#           mask_blur_ksize=(33, 33),
#           mask_threshold=66,
#           mask_blur_sigma=0)

# Clean the background
dse.clean(image_blur_ksize=(25, 25),
          image_threshold=100,
          image_blur_sigma=(0, 0),
          mask_blur_ksize=(7, 7),
          mask_threshold=dse.stack.info['Stack size'] * 0.85,
          mask_blur_sigma=0)

# Save mask
dse.save.binary(dse.mask, 'mask.bmp', increase_contrast=True)

# Filter
dse.kalman_filter(gain=0.75,
                  percent_var=0.05,
                  n_runs=3,
                  scheme=0)

# Save filtered pixel data as stack
dse.save.stack(dse.filtered_pixel_data, dse.stack, 'filtered_')

# Discretize core
dse.discretize(elem_side_length=1, name='mesh.msh')     # in mm

# Take measurements
measurements = dse.measure('measurements.csv', test=False)

# Stop timer and print time
elapsed_time = round(time.time() - start_time, 2)
print('\n============= Elapsed time: ', elapsed_time, ' sec =============')

# ======================================= Example of post processing data usage =======================================
# Evaluate and save porosity data
import numpy as np
poro = measurements['Mean gray scale value'] / 1000
np.save('output/poro.npy', poro)

# Wrap mesh with porosity
import meshio
mesh = meshio.read('output/physical_mesh.msh')
mesh.cell_data['wedge']['Porosity'] = poro
meshio.write('output/mesh_with_poro.vtk', mesh)

# Evaluate total volume and total pore volume (Specific for a particular example)
from math import pi
volume_analytical = 75 ** 2 / 4 * pi * 75
volume = np.sum(measurements['Volume'])
print('\nAnalytical volume:', volume_analytical, 'mm^3', '\nEvaluated volume:', volume, 'mm^3')

pv_analytical = volume_analytical * 0.278
pv = np.sum(measurements['Volume'] * poro)
print('\nAnalytical PV:', pv_analytical, '[ - ]', '\nEvaluated PV:', pv, '[ - ]')
# ======================================= Example of post processing data usage =======================================
