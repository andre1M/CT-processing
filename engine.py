from path_container import PathContainer
from discretizer import Discretizer
from stack import Stack
from saver import Saver

from math import sqrt
import numpy as np
import pydicom
import meshio
import cv2
import os


class DoubleStackEngine:
    def __init__(self):
        self.paths = None
        self.stack = None
        self.ref_stack = None
        self.pixel_data_diff = []
        self.mask = None
        self.clean_pixel_data = None
        self.filtered_pixel_data = None
        self.save = None
        self.discretizer = None

    def set(self, stack_dir: str, ref_stack_dir: str, resulting_stack_dir: str, output_dir: str):
        """
        Set paths to all required directories
        """
        self.paths = PathContainer(stack_dir, ref_stack_dir, resulting_stack_dir, output_dir)

    def read(self):
        """
        Read stacks and check their compatibility
        """
        # Initialize 'stack' and 'reference stack' objects
        self.stack = Stack(self.paths.stack)
        self.ref_stack = Stack(self.paths.ref_stack)

        # Check stack's metrics
        if self.stack.info['Stack size'] != self.ref_stack.info['Stack size']:
            raise RuntimeError('Stacks can\'t be processed; stack sizes don\'t match')
        elif self.stack.info['Resolution'] != self.ref_stack.info['Resolution']:
            raise RuntimeError('Stacks can\'t be processed; stack resolutions don\'t match')
        elif self.stack.info['Pixel spacing'] != self.ref_stack.info['Pixel spacing']:
            raise RuntimeError('Stacks can\'t be processed; stack pixel spacings don\'t match')
        elif self.stack.info['Slice thickness'] != self.ref_stack.info['Slice thickness']:
            raise RuntimeError('Stacks can\'t be processed; stack layer thicknesses don\'t match')
        else:
            self.stack.load()
            self.ref_stack.load()
            self.save = Saver(self.paths, self.stack.info)

    def clean(self, image_blur_ksize: tuple, image_threshold: float, mask_blur_ksize: tuple, mask_threshold: float,
              image_blur_sigma=0, mask_blur_sigma=0):
        """
        Find pixel data difference and clean the background with mask

        :param image_blur_ksize: height and width of Gaussian kernel; must contain exactly 2 odd integers.
        :param image_threshold: threshold for image to binary conversion.
        :param mask_blur_ksize: similar to image_blur_ksize but applied to mask to filter the residual noise.
        :param mask_threshold: threshold to introduce sharp edges to the mask (optional)
        :param image_blur_sigma: optional, see openCV documentation for Gaussian Blur.
        :param mask_blur_sigma: optional, see openCV documentation for Gaussian Blur.
        """
        self._find_difference()
        self._erase_background(image_blur_ksize, image_threshold, mask_blur_ksize, mask_threshold,
                               image_blur_sigma, mask_blur_sigma)

    # TODO: find a way to show two lines of pictures:
    #   1. mask, first, middle and last slices with background,
    #   2. empty image, first, middle and last slices without background.
    def tune(self, image_blur_ksize: tuple, image_threshold: float, mask_blur_ksize: tuple, mask_threshold: float,
             image_blur_sigma=0, mask_blur_sigma=0):
        """
        Tune parameters for further application of 'clean' method.

        :param image_blur_ksize: height and width of Gaussian kernel; must contain exactly 2 odd integers.
        :param image_threshold: threshold for image to binary conversion.
        :param mask_blur_ksize: similar to image_blur_ksize but applied to mask to filter the residual noise.
        :param mask_threshold: threshold to introduce sharp edges to the mask (optional)
        :param image_blur_sigma: optional, see openCV documentation for Gaussian Blur.
        :param mask_blur_sigma: optional, see openCV documentation for Gaussian Blur.
        """
        self._find_difference()
        self._erase_background(image_blur_ksize, image_threshold, mask_blur_ksize, mask_threshold,
                               image_blur_sigma, mask_blur_sigma)
        # Show results
        cv2.imshow("Mask (press any key to exit)", self.mask)
        cv2.waitKey(0)           # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image
        exit()

        # blank_image = np.zeros(self.mask.shape)
        # start = self.clean_stack[0]
        # middle = np.array(self.clean_stack[self.clean_stack.shape[0] // 2], dtype=np.uint16)
        # end = self.clean_stack[-1]
        #
        # cv2.imshow("Press any key to exit", np.hstack((self.mask * 2 ** 16, start, middle, end)))
        # cv2.waitKey(0)           # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image
        # exit()

    @staticmethod
    def _rescale_pixel_array(ct_slice):
        """
        Rescale pixel data using U = m * SV + b,
        where U -- output, m -- rescale slope, SV -- stored value and b -- rescale intercept.
        """
        return ct_slice.RescaleSlope * ct_slice.pixel_array + ct_slice.RescaleIntercept

    def _find_difference(self):
        """
        Calculate pixel data difference.
        """
        for i in range(self.stack.info['Stack size']):
            # Rescale pixel data
            ref_pixel_array = self._rescale_pixel_array(self.ref_stack.slices[i])
            pixel_array = self._rescale_pixel_array(self.stack.slices[i])
            # Calculate difference
            self.pixel_data_diff.append(ref_pixel_array - pixel_array)

    def _erase_background(self, image_blur_ksize, image_threshold, mask_blur_ksize, mask_threshold,
                          image_blur_sigma, mask_blur_sigma):
        """
        Erase background with mask
        """
        self._evaluate_mask(image_blur_ksize, image_threshold, mask_blur_ksize, mask_threshold,
                            image_blur_sigma, mask_blur_sigma)
        self.clean_pixel_data = np.multiply(np.array(self.pixel_data_diff), self.mask)

    def _evaluate_mask(self, image_blur_ksize, image_threshold, mask_blur_ksize, mask_threshold,
                       image_blur_sigma, mask_blur_sigma):
        """
        Evaluate the core representing mask
        """
        binary = []
        for i in range(self.stack.info['Stack size']):
            blurred = self._blur(self.pixel_data_diff[i], image_blur_ksize, image_blur_sigma)
            binary.append(self._convert_to_binary(blurred, image_threshold))

        mask = np.zeros((binary[0].shape[0], binary[0].shape[1]))
        start = int(self.stack.info['Stack size'] * 0.05)
        stop = int(self.stack.info['Stack size'] * 0.95)
        for i in range(start, stop):
            mask = mask + binary[i]
        mask = self._blur(mask, mask_blur_ksize, mask_blur_sigma)
        self.mask = self._convert_to_binary(mask, mask_threshold)

    @staticmethod
    def _blur(src, ksize, sigma):
        """
        Blur an image
        """
        if isinstance(sigma, tuple):
            if len(sigma) == 2:
                sigma_x = sigma[0]
                sigma_y = sigma[1]
                return cv2.GaussianBlur(src=src,
                                        ksize=ksize,
                                        sigmaX=sigma_x,
                                        sigmaY=sigma_y)
        elif isinstance(sigma, float) or isinstance(sigma, int):
            sigma_x = sigma
            return cv2.GaussianBlur(src=src,
                                    ksize=ksize,
                                    sigmaX=sigma_x)
        else:
            raise ValueError('\'sigma\' variable must contain either 1 or 2 values')

    @staticmethod
    def _convert_to_binary(src, image_threshold, max_val=1):
        _, binary = cv2.threshold(src=src,
                                  thresh=image_threshold,
                                  maxval=max_val,
                                  type=cv2.THRESH_BINARY)
        return binary

    # TODO: finish attributes description
    def kalman_filter(self, gain, percent_var, n_runs: int, scheme=0):
        """
        Kalman stack filter interface

        :param gain:
        :param percent_var:
        :param n_runs: number of runs the filter would go through the stack
        :param scheme: 0 -- alternate between forward and reverse filtering directions each run
                       1 -- forward filtering direction
                       2 -- reverse filtering direction
        """
        i = 0
        self.filtered_pixel_data = self.clean_pixel_data
        if scheme == 0:
            reverse = False
            while i in range(n_runs):
                self._kalman(gain, percent_var, reverse)
                i += 1
                reverse = not reverse

        elif scheme == 1:
            reverse = False
            while i in range(n_runs):
                self._kalman(gain, percent_var, reverse)
                i += 1

        elif scheme == 2:
            reverse = True
            while i in range(n_runs):
                self._kalman(gain, percent_var, reverse)
                i += 1
        else:
            raise ValueError('\'scheme\' must have an integer value of either 0, 1 or 2')

    def _kalman(self, gain, percent_var, reverse):
        """
        Kalman stack filter
        """
        if reverse:
            self.filtered_pixel_data = np.flip(self.filtered_pixel_data, axis=0)

        # Copy last slice to the end of the stack
        self.filtered_pixel_data = np.concatenate((self.filtered_pixel_data, self.filtered_pixel_data[-1:, :, :]))

        # Set up priors
        noise_var = percent_var
        predicted = self.filtered_pixel_data[0, :, :]
        predicted_var = noise_var

        for i in range(0, self.filtered_pixel_data.shape[0] - 1):
            observed = self.filtered_pixel_data[i + 1, :, :]
            kalman = np.divide(predicted_var, np.add(predicted_var, noise_var))
            corrected = gain * predicted + (1 - gain) * observed + np.multiply(kalman, np.subtract(observed, predicted))
            corrected_var = np.multiply(predicted_var, (1 - kalman))

            predicted_var = corrected_var
            predicted = corrected
            self.filtered_pixel_data[i + 1, :, :] = corrected

        self.filtered_pixel_data = self.filtered_pixel_data[:-1, :, :]
        if reverse:
            self.filtered_pixel_data = np.flip(self.filtered_pixel_data, axis=0)

    # TODO: finish method (optional, for later versions)
    def gaussian_blur(self):
        pass

    def discretize(self, elem_side_length: float, name: str):
        """
        Discretize core and save mesh file

        :param elem_side_length: side length of the element at the core boundary
        :param name: name of the output mesh file, must have .msh extension
        """
        self.discretizer = Discretizer(self.stack.info, self.mask, self.paths, self.save)
        self.discretizer.evaluate_geometry(elem_side_length)
        self.discretizer.mesh(name)

    def measure(self, results_name):
        """
        Take measurements with Fiji
        """
        self._generate_fiji_macro(results_name)
        self._run_fiji()

    # TODO: finish method
    def _generate_fiji_macro(self, results_name, processing=True):
        mesh = self.discretizer.get_fiji_mesh()
        elements = mesh.cells['triangle']
        nodes = mesh.points
        string = 'run("Image Sequence...", "open=%s sort");\n' % self.paths.resulting_stack
        string += 'run("Gaussian Blur...", "sigma=2 stack");\n'
        for i in range(len(elements)):
            x1 = nodes[int(elements[i, 0]), 1]
            y1 = nodes[int(elements[i, 0]), 0]
            x2 = nodes[int(elements[i, 1]), 1]
            y2 = nodes[int(elements[i, 1]), 0]
            x3 = nodes[int(elements[i, 2]), 1]
            y3 = nodes[int(elements[i, 2]), 0]
            string += 'makePolygon(%d,%d,%d,%d,%d,%d);\nroiManager("Add");\n' % (x1, y1, x2, y2, x3, y3)
        if processing:
            for i in range(self.stack.size):
                string += 'roiManager("Measure");\nrun("Next Slice [>]");\n'
            string += 'saveAs("Measurements", "%s")\n' % (self.paths.output + results_name)
            string += 'run("Quit");'
        file = open(self.paths + 'macro.ijm', "w+")
        file.write(string)
        file.close()

    def _run_fiji(self):
        pass







    def correct_mesh(self, mesh_file):
        """Rounds mesh coordinates for nodes and convert y-axis coordinates;
        required because Fiji creates nodes only at pixel locations"""
        mesh = meshio.read(self.output_dir + mesh_file)
        # Get node coordinates and adapt them to the Fiji coordinate system
        mesh.points[:, 0] = np.around(mesh.points[:, 0])
        mesh.points[:, 1] = np.around(mesh.points[:, 1])
        meshio.write_points_cells(self.output_dir + 'corrected_' + mesh_file, mesh.points,
                                  {'wedge': mesh.cells['wedge']})
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
        os.system('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx -macro %s' % (self.output_dir + macro))

    def get_measurements(self, measurements_file, mesh):
        """Extract mean grey level and area values per mesh element from Fiji output;
        available only when resulting .csv file is supplied"""
        elements = meshio.read(self.output_dir + mesh).cells['triangle']
        measurements_raw_data = pd.read_csv(self.output_dir + measurements_file)
        poro_per_element = measurements_raw_data['Mean'].values / 1000
        area_per_element = measurements_raw_data['Area'].values

        if len(poro_per_element) == self.stack.size * elements.shape[0] and \
                len(area_per_element) == self.stack.size * elements.shape[0]:
            print('\nProcessing complete successfully. No data is lost')
        else:
            raise ValueError('\nProcessing complete with errors. Some data is lost')
        return {'poro': poro_per_element,
                'area': area_per_element}

    def wrap_mesh(self, measurements, mesh, mesh_2d):
        mesh_3d = meshio.read(self.output_dir + mesh)
        mesh_2d = meshio.read(self.output_dir + mesh_2d)
        elements = mesh_2d.cells['triangle']
        poro_3d_ordered = np.zeros(len(elements) * (self.stack.size - 1))
        vol_3d_ordered = np.zeros(len(elements) * (self.stack.size - 1))
        poro_3d = np.zeros(len(elements) * (self.stack.size - 1))

        for i in range(len(poro_3d)):
            poro_3d[i] = (measurements['poro'][i] + measurements['poro'][len(elements) + i]) / 2

        for i in range(len(elements)):
            poro_3d_ordered[i * (self.stack.size - 1):(i + 1) * (self.stack.size - 1)] = poro_3d[i::len(elements)]
            vol_3d_ordered[i * (self.stack.size - 1):(i + 1) * (self.stack.size - 1)] = \
                measurements['area'][i] * self.layer_thickness * 1e-9

        mesh_3d.cell_data['wedge']['porosity'] = poro_3d_ordered
        mesh_3d.cell_data['wedge']['volume'] = vol_3d_ordered
        meshio.write(self.output_dir + 'mesh_final.vtk', mesh_3d)
        np.save(self.output_dir + 'poro.npy', poro_3d_ordered)

    def get_physical_geometry(self, name, mesh_contour):
        string = '//+\nSetFactory("OpenCASCADE");\nlc = 0;\n'
        for k in range(len(mesh_contour)):
            i, j = mesh_contour[k]
            string += 'Point(%d) = {0, %f, %f, lc};\n' % (k + 1, i, j)
        for k in range(len(mesh_contour)):
            if k == len(mesh_contour) - 1:
                string += 'Line(%d) = {%d, %d};\n' % (k + 1, k + 1, 1)
            else:
                string += 'Line(%d) = {%d, %d};\n' % (k + 1, k + 1, k + 2)
        string += 'Line Loop(%d) = {' % (len(mesh_contour) + 1)
        for k in range(len(mesh_contour) - 1):
            string += '%d, ' % (k + 1)
        string += '%d};\n' % (len(mesh_contour))
        string += 'Plane Surface(%d) = {%d};\n' % (len(mesh_contour) + 2, len(mesh_contour) + 1)
        string += '\nExtrude{%f, 0, 0}\n' % (self.layer_thickness * self.stack.size / 1000)
        string += '{\nSurface{%d}; Layers{%d}; Recombine;\n}' % (len(mesh_contour) + 2, self.stack.size - 1)
        # string += '{\nSurface{%d}; Layers{%d};\n}' % (len(mesh_contour) + 2, self.stack.size - 1)
        string += '\n\nPhysical Volume("Inner Volume") = {1};\n'
        file = open(self.output_dir + name, "w+")
        file.write(string)
        file.close()

    def generate_physical_mesh(self, mesh_contour):
        physical_geo = 'physical.geo'
        self.get_physical_geometry(physical_geo, mesh_contour)
        self.generate_3d_mesh(physical_geo, 'physical_mesh.msh')
        mesh = meshio.read('output/physical_mesh.msh')
        mesh.points[:, 1:] = mesh.points[:, 1:] * self.pixel_spacing / 1000
        # mesh.points[:, 0] = mesh.points[:, -1] * self.layer_thickness * self.stack.size / 1000
        meshio.write(self.output_dir + 'physical_mesh_corrected.msh', mesh)

    @staticmethod
    def get_adjusted_mesh_contour(contour, elem_size):
        # while True:
        mesh_contour = [[contour[0][1], contour[0][0]]]
        i = 0
        while i < len(contour) - 1:
            x = contour[i][0]
            y = contour[i][1]
            for j in range(i + 1, len(contour)):
                x_n = contour[j][0]
                y_n = contour[j][1]
                dist = sqrt(
                    abs(y - y_n) ** 2 + abs(x - x_n) ** 2
                )
                if dist >= elem_size:
                    mesh_contour.append([y_n, x_n])
                    i = j
                    break
                elif j == len(contour) - 1:
                    i = j
                    break
        mesh_contour.pop(0)
        return mesh_contour
