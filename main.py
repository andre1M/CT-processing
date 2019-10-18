from engine import ImageProcessingEngine
import time

# Start timer
start_time = time.time()

# Initialize engine
ipen = ImageProcessingEngine('output', 'filtered', 'brine', 'dry')

# Subtract stack pixel data from reference stack pixel data
subtracted_pixel_data = ipen.subtract()

# Create and save mask
mask = ipen.get_mask(subtracted_pixel_data, 65, 59, 0.15)
ipen.save_binary_as_image(mask, 'mask.bmp')

# Delete all information beyond core region
masked_pixel_data = ipen.apply_mask(subtracted_pixel_data, mask, correction=1024)

# Filter resulting stack
transition_pixel_data = ipen.kalman_stack_filter(masked_pixel_data, 0.75, 0.05, reverse=False)
filtered_pixel_data = ipen.kalman_stack_filter(transition_pixel_data, 0.75, 0.05, reverse=True)

# Save filtered stack
ipen.save_stack(filtered_pixel_data, 'filtered', 'IMA')

# Find mask contour
contour = ipen.get_mask_contour(mask)

# Save contour as image
ipen.draw_contour(contour, mask.shape, 'contour.bmp')

# Get mesh contour
mesh_contour = ipen.get_mesh_contour(contour, 18)

# Save mesh contour as image
ipen.draw_contour(mesh_contour, mask.shape, 'mesh_contour.bmp')

# Write Gmsh geometry file
ipen.write_gmsh_geometry(mesh_contour, 'core.geo')

# Generate 3D mesh
ipen.generate_3d_mesh('core.geo', 'mesh.msh')

# Correct mesh
corrected_mesh = ipen.correct_mesh('mesh.msh')

# Extract 2D mesh section for Fiji from 3D corrected mesh
ipen.get_fiji_mesh(corrected_mesh, 'fiji.msh')

# Write Fiji marcos
ipen.generate_macro('fiji.msh', 'measurements.csv', 'macro.ijm', with_processing=False)

# Run Fiji macros
ipen.run_fiji('macro.ijm')

# mesh_data = ipen.collect_mesh_data('measurements.csv')
# mesh_with_data = ipen.wrap_mesh_with_data(mesh_data, 'mesh.msh')
# ipen.write_mesh_with_data(mesh_with_data, 'vtk')

# # ====================== 9. Generate porosity map ======================
# mean_per_element = engine.filter_results(measurements_file, path.output, stack.size, elements)
# poro_3d = engine.get_porosity(mean_per_element)
# engine.write_poro2mesh(poro_3d, nodes, elements, path.output, corrected_mesh_name)

# Stop timer and print time
elapsed_time = round(time.time() - start_time, 2)
print('\n============= Elapsed time: ', elapsed_time, ' sec =============')
