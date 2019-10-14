from PIL import Image
import pandas as pd
import numpy as np
import pydicom
import meshio
import cv2
import os
import re


class ImageProcessingEngine:
    def __init__(self, output_dir, resulting_stack_dir, ref_stack_dir, stack_dir):
        # Required paths
        self.output_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + output_dir + '/'
        self.resulting_stack_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + resulting_stack_dir + '/'

        # Initialize stack and reference stack objects
        self.stack = Stack(os.path.dirname(os.path.abspath(__file__)) + '/' + stack_dir + '/')
        self.ref_stack = Stack(os.path.dirname(os.path.abspath(__file__)) + '/' + ref_stack_dir + '/')

        # Check stack sizes
        if self.stack.size != self.ref_stack.size:
            raise ValueError('Stacks can\'t be processed; stack sizes don\'t match')

        # Get essential information about images (should be the same for both stacks)
        if self.stack.read_slice(0).PixelSpacing[0] == self.ref_stack.read_slice(0).PixelSpacing[0]:
            self.pixel_spacing = self.stack.read_slice(0).PixelSpacing[0]
        else:
            raise ValueError('Stacks can\'t be processed due to different pixel spacing')

        if self.stack.read_slice(0).SliceThickness == self.ref_stack.read_slice(0).SliceThickness:
            self.layer_thickness = self.stack.read_slice(0).SliceThickness
        else:
            raise ValueError('Stacks can\'t be processed due to different layer thickness')

    def subtract(self):
        return self.ref_stack.pixel_data - self.stack.pixel_data

    @staticmethod
    def blur(image, ksize):
        """Blurs an image"""
        return cv2.GaussianBlur(src=image,
                                ksize=(ksize, ksize),
                                sigmaX=0)

    def get_binary(self, image, threshold, ksize):
        """Converts image to binary"""
        _, binary = cv2.threshold(src=self.blur(image, ksize),
                                  thresh=threshold,
                                  maxval=1,
                                  type=cv2.THRESH_BINARY)
        return binary

    def save_binary_as_image(self, binary_pixel_array, name, increase_contrast=True):
        """Saves data array as a binary image"""
        data = np.array(binary_pixel_array)
        if increase_contrast:
            data[data > 0] = 255
        try:
            im = Image.fromarray(data)
            im = im.convert('L')
            im.save(self.output_dir + name)
        except TypeError:
            data = cv2.convertScaleAbs(data)
            im = Image.fromarray(data)
            im = im.convert('L')
            im.save(self.output_dir + name)

    def get_mask(self, stack_pixel_data, threshold, ksize, app_range):
        """Locates core in an image and creates mask"""
        binary = np.zeros(stack_pixel_data.shape)
        for i in range(binary.shape[0]):
            binary[i] = self.get_binary(stack_pixel_data[i], threshold, ksize)

        if isinstance(app_range, float):
            approximation_range = [int(binary.shape[0] * app_range),
                                   int(binary.shape[0] - binary.shape[0] * app_range)]
        elif isinstance(app_range, list) and len(app_range) == 2:
            approximation_range = [int(binary.shape[0] * app_range[0]),
                                   int(binary.shape[0] - binary.shape[0] * app_range[1])]
        else:
            raise ValueError('app_range determines which fraction of slices will be excluded from the evaluation;'
                             'Must be either a single value or two values (for both sides of a core)')

        mask = np.ones((binary.shape[1], binary.shape[2]))
        for i in range(approximation_range[0], approximation_range[1]):
            mask = mask * binary[i]
        for i in range(approximation_range[0], approximation_range[1]):
            mask = cv2.bitwise_and(mask, binary[i])
        mask[mask == -0] = 0
        return mask

    def filter_mask(self, mask, threshold, enlarge=0):
        """Filters mask: correct shape, eliminate black spots withing a core region"""
        borders = self.locate_borders(mask, threshold)
        center, radius = self.evaluate_geometry(borders)
        cv2.circle(mask, (center[1], center[0]), radius + enlarge, 1, thickness=-1)
        mask[mask == -0] = 0
        return mask

    @staticmethod
    def locate_borders(mask, threshold):
        """Locates borders of a circular core from its mask"""
        top, right, bottom, left = 0, 0, 0, 0
        i = 0
        while top == 0:
            if np.count_nonzero(mask[i, :] > 0) > threshold:
                top = i
            i += 1

        i = mask.shape[1] - 1
        while right == 0:
            if np.count_nonzero(mask[:, i] > 0) > threshold:
                right = i
            i -= 1

        i = mask.shape[0] - 1
        while bottom == 0:
            if np.count_nonzero(mask[i, :] > 0) > threshold:
                bottom = i
            i -= 1

        i = 0
        while left == 0:
            if np.count_nonzero(mask[:, i] > 0) > threshold:
                left = i
            i += 1
        return {'top': top, 'right': right, 'bottom': bottom, 'left': left}

    @staticmethod
    def evaluate_geometry(borders):
        """Evaluates geometry of a core"""
        radius = int(round((borders['bottom'] - borders['top'] + borders['right'] - borders['left']) / 4))
        center = [int(round((borders['top'] + borders['bottom']) / 2)),
                  int(round((borders['left'] + borders['right']) / 2))]
        return center, radius

    @staticmethod
    def apply_mask(stack_pixel_data, mask, correction=0):
        """Deletes any information beyond a core"""
        clean_stack_pixel_data = np.zeros(stack_pixel_data.shape)
        for i in range(stack_pixel_data.shape[0]):
            clean_stack_pixel_data[i] = stack_pixel_data[i] * mask
        clean_stack_pixel_data[clean_stack_pixel_data == -0] = 0
        return clean_stack_pixel_data + correction

    @staticmethod
    def kalman_stack_filter(stack_pixel_data, gain, percent_var, reverse=False):
        """Applies Kalman stack filter"""
        if reverse:
            stack_pixel_data = np.flip(stack_pixel_data, axis=0)

        # Set up priors
        tmp = np.ones((stack_pixel_data.shape[1], stack_pixel_data.shape[2]))
        predicted = stack_pixel_data[0, :, :]
        predicted_var = tmp * percent_var
        noise_var = predicted_var

        # Copy last slice to the end of the stack
        stack_pixel_data = np.concatenate((stack_pixel_data, stack_pixel_data[-1:, :, :]))

        for i in range(0, stack_pixel_data.shape[0] - 1):
            observed = stack_pixel_data[i + 1, :, :]
            kalman = np.divide(predicted_var, np.add(predicted_var, noise_var))
            corrected = gain * predicted + (1.0 - gain) * observed + np.multiply(kalman,
                                                                                 np.subtract(observed, predicted))
            corrected_var = np.multiply(predicted_var, np.subtract(tmp, kalman))

            predicted_var = corrected_var
            predicted = corrected
            stack_pixel_data[i, :, :] = corrected

        stack_pixel_data = stack_pixel_data[:-1, :, :]
        if reverse:
            stack_pixel_data = np.flip(stack_pixel_data, axis=0)
        return stack_pixel_data

    def save_stack(self, data, naming_convention, extension):
        """Saves data array as image without scaling"""
        for i in range(data.shape[0]):
            template_slice = self.stack.read_slice(i)
            template_slice.PixelData = np.array(data[i], dtype=template_slice.pixel_array.dtype).tobytes()
            pydicom.filewriter.dcmwrite(self.resulting_stack_dir + naming_convention + str(i + 1) + '.' + extension,
                                        template_slice)

    @staticmethod
    def get_mask_contour(mask):
        """Locates all contour points of a mask"""
        im = np.array(mask, dtype='uint8')
        contour, _ = cv2.findContours(image=im,
                                      mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
        contour_list = []
        for i in range(len(contour[0])):
            contour_list.append([contour[0][i, 0, 1], contour[0][i, 0, 0]])
        return contour_list

    def draw_contour(self, contour, resolution, name):
        image = np.zeros((resolution[1], resolution[0]))
        for i, j in contour:
            image[i, j] = 1
        self.save_binary_as_image(image, name)
        return image

    def write_gmsh_geometry(self, contour, name):
        """Generates a set of commands for Gmsh to create an object for further meshing"""
        string = '//+\nSetFactory("OpenCASCADE");\nlc = 0;\n'
        for k in range(len(contour)):
            i, j = contour[k]
            string += 'Point(%d) = {%d, %d, 0, lc};\n' % (k + 1, j, -i)
        for k in range(len(contour)):
            if k == len(contour) - 1:
                string += 'Line(%d) = {%d, %d};\n' % (k + 1, k + 1, 1)
            else:
                string += 'Line(%d) = {%d, %d};\n' % (k + 1, k + 1, k + 2)
        string += 'Line Loop(%d) = {' % (len(contour) + 1)
        for k in range(len(contour) - 1):
            string += '%d, ' % (k + 1)
        string += '%d};\n' % (len(contour))
        string += 'Plane Surface(%d) = {%d};\n' % (len(contour) + 2, len(contour) + 1)
        string += '\nExtrude{0, 0, %.1f}\n' % (round(self.layer_thickness / self.pixel_spacing, 1) *
                                               (self.stack.size - 1))
        string += '{\nSurface{%d}; Layers{%d}; Recombine;\n}' % (len(contour) + 2, self.stack.size - 1)
        file = open(self.output_dir + name, "w+")
        file.write(string)
        file.close()

    @staticmethod
    def get_mesh_contour(contour, grid_side_length):
        mesh_contour = []
        for i in range(0, len(contour) - grid_side_length, grid_side_length):
            mesh_contour.append([contour[i][1], contour[i][0]])
        return mesh_contour

    def generate_3d_mesh(self, geo_file, name):
        """Launch Gmsh and generate mesh for the defined geometry"""
        os.system('gmsh %s -3 -o %s' % (self.output_dir + geo_file, self.output_dir + name))

    def correct_mesh(self, mesh_file):
        """Rounds mesh coordinates for nodes and convert y-axis coordinates;
        required because Fiji creates nodes only at pixel locations"""
        mesh = meshio.read(self.output_dir + mesh_file)
        # Get node coordinates and adapt them to the Fiji coordinate system
        mesh.points[:, 0] = np.around(mesh.points[:, 0])
        mesh.points[0, :] = np.around(mesh.points[0, :])
        # Convert y-axes (zero point is at the top left corner)
        mesh.points[mesh.points < 0] *= -1
        meshio.write_points_cells(self.output_dir + 'corrected_' + mesh_file, mesh.points, {'wedge': mesh.cells['wedge']})
        return 'corrected_' + mesh_file

    def get_fiji_mesh(self, mesh_name, name):
        """Extracts 2D mesh slice from 3D mesh file"""
        mesh = meshio.read(self.output_dir + mesh_name)
        indices = np.array([])
        nodes = np.array([[0, 0, 0]])
        # Get all nodes that belong to the first layer with their indices
        for i in range(mesh.points.shape[0]):
            if mesh.points[i, 2] == 0:
                indices = np.append(indices, i)
                entity = mesh.points[i, :][np.newaxis]
                nodes = np.concatenate((nodes, entity), axis=0)
        nodes = nodes[1:, :]
        # Get all triangles that belong to the first layer
        elements = np.array([[0, 0, 0]])
        for i in range(mesh.cells['wedge'].shape[0]):
            c = 0
            triangle = np.zeros(3)
            for j in range(mesh.cells['wedge'].shape[1]):
                if mesh.cells['wedge'][i, j] in indices:
                    triangle[c] = mesh.cells['wedge'][i, j]
                    c += 1
            if c == 3:
                new_triangle = np.zeros(3)[np.newaxis]
                for k in range(c):
                    new_triangle[0, k] = np.where(indices == triangle[k])[0]
                elements = np.concatenate((elements, new_triangle), axis=0)
        elements = elements[1:, :]
        meshio.write_points_cells(self.output_dir + name, nodes, {'triangle': elements.astype(int)})

    def generate_macro(self, fiji_mesh, measurements_file, name, with_processing=True):
        """Generate Fiji macro commands to discretize and process stack"""
        mesh = meshio.read(self.output_dir + fiji_mesh)
        elements = mesh.cells['triangle']
        nodes = mesh.points
        string = 'run("Image Sequence...", "open=%s sort");\n' % self.resulting_stack_dir
        string += 'run("Gaussian Blur...", "sigma=2 stack");\n'
        for i in range(len(elements)):
            x1 = nodes[int(elements[i, 0]), 1]
            y1 = nodes[int(elements[i, 0]), 0]
            x2 = nodes[int(elements[i, 1]), 1]
            y2 = nodes[int(elements[i, 1]), 0]
            x3 = nodes[int(elements[i, 2]), 1]
            y3 = nodes[int(elements[i, 2]), 0]
            string += 'makePolygon(%d,%d,%d,%d,%d,%d);\nroiManager("Add");\n' % (x1, y1, x2, y2, x3, y3)
        if with_processing:
            for i in range(self.stack.size):
                string += 'roiManager("Measure");\nrun("Next Slice [>]");\n'
            string += 'saveAs("Measurements", "%s")\n' % (self.output_dir + measurements_file)
            string += 'run("Quit");'
        file = open(self.output_dir + name, "w+")
        file.write(string)
        file.close()

    def run_fiji(self, macro):
        os.system('ImageJ-macosx -macro %s' % (self.output_dir + macro))

    # @staticmethod
    # def filter_results(measurements_file, output_dir, stack_size, elements):
    #     """Extracts mean grey level values per mesh element from Fiji output;
    #     available only when resulting .csv file is supplied"""
    #     measurements_raw_data = pd.read_csv(output_dir + measurements_file)
    #     measurements_filtered = np.zeros((stack_size, 1, elements.shape[0]))
    #     for i in range(stack_size):
    #         measurements_filtered[i, 0, :] = \
    #             measurements_raw_data.iloc[i * elements.shape[0]:(i + 1) * elements.shape[0]]['Mean']
    #     mean_per_element = measurements_filtered[:]
    #     if measurements_raw_data.shape[0] == stack_size * elements.shape[0]:
    #         print('\nProcessing complete successfully. No data is lost')
    #         return mean_per_element
    #     else:
    #         raise ValueError('\nProcessing complete with errors. Some data is lost')


class Stack:
    def __init__(self, path):
        self.path = path
        self.names = self.read_file_names()
        self.size = len(self.names)
        self.resolution = self.get_resolution()
        self.pixel_data = self.collect_slices()

    def read_file_names(self):
        """Scans folder with CT and gets slice names (.IMA format)"""
        files = os.listdir(self.path)
        files.sort()
        extension = re.compile('.*IMA')
        images = list(filter(extension.match, files))
        return images

    def get_resolution(self):
        """Determines resolution of an image (in pixels)"""
        instance = self.read_slice(0)
        resolution = {'y': instance.Rows,
                      'x': instance.Columns}
        return resolution

    def read_slice(self, index):
        """Reads a single slice from stack"""
        slice_data = pydicom.dcmread(self.path + self.names[index])
        return slice_data

    def collect_slices(self):
        """Collects all slices into a single array"""
        matrix = np.zeros((self.size, self.resolution['y'], self.resolution['x']))
        for i in range(self.size):
            matrix[i] = self.read_slice(i).pixel_array
        return np.array(matrix)





    # @staticmethod
    # def generate_macro(nodes, elements, measurements_file, stack_size, filtered_dir, output_dir, with_processing=True):
    #     """Generate Fiji macro commands to discretize and process stack"""
    #     string = 'run("Image Sequence...", "open=%s sort");\n' % filtered_dir
    #     string += 'run("Gaussian Blur...", "sigma=2 stack");\n'
    #     for i in range(len(elements)):
    #         x1 = nodes[int(elements[i, 0]), 0]
    #         y1 = nodes[int(elements[i, 0]), 1]
    #         x2 = nodes[int(elements[i, 1]), 0]
    #         y2 = nodes[int(elements[i, 1]), 1]
    #         x3 = nodes[int(elements[i, 2]), 0]
    #         y3 = nodes[int(elements[i, 2]), 1]
    #         string += 'makePolygon(%d,%d,%d,%d,%d,%d);\nroiManager("Add");\n' % (x1, y1, x2, y2, x3, y3)
    #     if with_processing:
    #         for i in range(stack_size):
    #             string += 'roiManager("Measure");\nrun("Next Slice [>]");\n'
    #         string += 'saveAs("Measurements", "%s")\n' % (output_dir + measurements_file)
    #         string += 'run("Quit");'
    #     return string
    #
    # @staticmethod
    # def write_macro(macro, macro_file, output_dir):
    #     """Writes Fiji macros"""
    #     file = open(output_dir + macro_file, "w+")
    #     file.write(macro)
    #     file.close()
    #
    # @staticmethod
    # def run_macro(macro_file, output_dir, headless=True):
    #     """Runs Fiji macros"""
    #     if headless:
    #         os.system('ImageJ-macosx --headless -macro %s' % (output_dir + macro_file))
    #     else:
    #         os.system('ImageJ-macosx -macro %s' % (output_dir + macro_file))
    #
    # @staticmethod
    # def get_porosity(mean_grey_value):
    #     """Generates porosity values for 3D as simple average between two adjacent slices"""
    #     poro_per_roi = mean_grey_value / 1000
    #     poro_per_roi[poro_per_roi < 0] = 0
    #     poro_3d = np.zeros((mean_grey_value.shape[0] - 1) * mean_grey_value.shape[2])
    #     for i in range(mean_grey_value.shape[0] - 1):
    #         for j in range(mean_grey_value.shape[2]):
    #             poro_3d[i * mean_grey_value.shape[2] + j] = (poro_per_roi[i, 0, j] + poro_per_roi[i + 1, 0, j]) / 2
    #     return poro_3d
    #
    # @staticmethod
    # def filter_results(measurements_file, output_dir, stack_size, elements):
    #     """Extracts mean grey level values per mesh element from Fiji output;
    #     available only when resulting .csv file is supplied"""
    #     measurements_raw_data = pd.read_csv(output_dir + measurements_file)
    #     measurements_filtered = np.zeros((stack_size, 1, elements.shape[0]))
    #     for i in range(stack_size):
    #         measurements_filtered[i, 0, :] = \
    #             measurements_raw_data.iloc[i * elements.shape[0]:(i + 1) * elements.shape[0]]['Mean']
    #     mean_per_element = measurements_filtered[:]
    #     if measurements_raw_data.shape[0] == stack_size * elements.shape[0]:
    #         print('\nProcessing complete successfully. No data is lost')
    #         return mean_per_element
    #     else:
    #         raise ValueError('\nProcessing complete with errors. Some data is lost')
    #
    # @staticmethod
    # def write_poro2mesh(poro_3d, nodes, elements, output_dir, mesh_name):
    #     """Writes porosity to a mesh file"""
    #     mesh_3d = meshio.read(output_dir + mesh_name)
    #     poro_3d_ordered = np.array([])
    #     coordinates = np.zeros((mesh_3d.cells['wedge'].shape[1], mesh_3d.points.shape[1]))
    #     horizontal_step = elements.shape[0]
    #     vertical_step = int(poro_3d.shape[0] / horizontal_step)
    #     for i in range(0, mesh_3d.cells['wedge'].shape[0], vertical_step):
    #         for j in range(mesh_3d.cells['wedge'].shape[1]):
    #             coordinates[j, :] = mesh_3d.points[mesh_3d.cells['wedge'][i, j]]
    #         for k in range(horizontal_step):
    #             if (coordinates[:3, :] == nodes[elements[k].astype(int)][:, :]).all():
    #                 poro_3d_ordered = np.append(poro_3d_ordered, poro_3d[k::horizontal_step])
    #                 break
    #     np.savetxt(output_dir + 'poro.csv', poro_3d_ordered, delimiter=",")
    #     mesh_3d.cell_data['wedge']['gmsh:physical'] = poro_3d_ordered
    #     mesh_3d.cell_data['wedge']['gmsh:geometrical'] = poro_3d_ordered
    #     meshio.write(output_dir + 'mesh_final.vtk', mesh_3d)
