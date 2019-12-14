from path_container import PathContainer
from discretizer import Discretizer
from stack import Stack
from saver import Saver

import pandas as pd
import numpy as np
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

        # ==================================== Prototype ====================================
        # blank_image = np.zeros(self.mask.shape)
        # start = self.clean_stack[0]
        # middle = np.array(self.clean_stack[self.clean_stack.shape[0] // 2], dtype=np.uint16)
        # end = self.clean_stack[-1]
        #
        # cv2.imshow("Press any key to exit", np.hstack((self.mask * 2 ** 16, start, middle, end)))
        # cv2.waitKey(0)           # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image
        # exit()
        # ==================================== Prototype ====================================

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

        # To avoid anomalies in the mask that may be introduced by close to core ends slices
        start = int(self.stack.info['Stack size'] * 0.05)
        stop = int(self.stack.info['Stack size'] * 0.95)

        mask = np.zeros((binary[0].shape[0], binary[0].shape[1]))
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
        self.discretizer.evaluate_physical_mesh(name)

    def measure(self, results_name: str, test=False):
        """
        Take measurements with Fiji and perform postprocessing

        :param results_name: file name to be used to save Fiji readings
        :param test: if True no processing step will be added to Fiji macro; measurements will not be taken;
            script will be paused until Fiji is manually closed
        """
        mesh = self._generate_fiji_macro(results_name, test)
        self._run_fiji()
        measurements = self._rescale_measurements(results_name)
        measurements_resorted = self._resort_measurements(measurements, mesh)
        return self._evaluate_3d(measurements_resorted, mesh)

    def _generate_fiji_macro(self, results_name, test):
        """
        Generate macro instructions file for Fiji
        """
        mesh = self.discretizer.get_fiji_mesh()
        elements = mesh.cells['triangle']
        nodes = mesh.points
        string = 'run("Image Sequence...", "open=%s sort");\n' % self.paths.resulting_stack
        string += 'run("Gaussian Blur...", "sigma=3 stack");\n'
        for i in range(len(elements)):
            x1 = nodes[int(elements[i, 0]), 2]
            y1 = nodes[int(elements[i, 0]), 1]
            x2 = nodes[int(elements[i, 1]), 2]
            y2 = nodes[int(elements[i, 1]), 1]
            x3 = nodes[int(elements[i, 2]), 2]
            y3 = nodes[int(elements[i, 2]), 1]
            string += 'makePolygon(%d,%d,%d,%d,%d,%d);\nroiManager("Add");\n' % (x1, y1, x2, y2, x3, y3)
        if not test:
            for i in range(self.stack.info['Stack size']):
                string += 'roiManager("Measure");\nrun("Next Slice [>]");\n'
            string += 'saveAs("Measurements", "%s")\n' % (self.paths.output + results_name)
            string += 'run("Quit");'
        file = open(self.paths.output + 'macro.ijm', "w+")
        file.write(string)
        file.close()
        return mesh

    def _run_fiji(self):
        """
        Launch Fiji to process macro
        """
        os.system('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx -macro %s' % (self.paths.output + 'macro.ijm'))

    def _rescale_measurements(self, name):
        """
        Read and rescale Fiji readings
        """
        df = pd.read_csv(self.paths.output + name)
        mean_gray_scale = (
                self.stack.slices[0].RescaleSlope * df['Mean'].values - self.stack.slices[0].RescaleIntercept
        )
        area = df['Area'].values
        return {'Mean gray scale value': mean_gray_scale, 'Area': area}

    def _resort_measurements(self, measurements, mesh):
        """
        Resort measurements according mesh cells ordering in 3D
        """
        elems_per_slice = mesh.cells['triangle'].shape[0]
        measurements_resorted = {}
        for key, data in measurements.items():
            resorted = np.zeros(data.shape)
            for i in range(elems_per_slice):
                resorted[i * self.stack.info['Stack size']:(i + 1) * self.stack.info['Stack size']] = \
                    data[i::elems_per_slice]
            measurements_resorted[key] = resorted
        return measurements_resorted

    def _evaluate_3d(self, measurements, mesh):
        """
        Evaluate properties for 3D elements
        """
        measurements_3d = {}
        elems_per_slice = mesh.cells['triangle'].shape[0]
        vols_per_stack = self.stack.info['Stack size'] - 1
        for key, data in measurements.items():
            evaluated = np.zeros(data.shape[0] - elems_per_slice)
            for i in range(elems_per_slice):
                evaluated[i * vols_per_stack:(i + 1) * vols_per_stack] = \
                    (data[i * self.stack.info['Stack size']:(i + 1) * self.stack.info['Stack size'] - 1] +
                     data[i * self.stack.info['Stack size'] + 1:(i + 1) * self.stack.info['Stack size']]) / 2
            measurements_3d[key] = evaluated
        return measurements_3d
