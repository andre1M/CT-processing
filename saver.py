from PIL import Image
from copy import deepcopy
import numpy as np
import pydicom
import cv2


class Saver:
    def __init__(self, paths, info):
        self.paths = paths
        self.stack_info = info

    def binary(self, binary_pixel_array, name, increase_contrast=True):
        """
        Save pixel array as a binary image
        """
        data = np.array(binary_pixel_array)
        if increase_contrast:
            data[data > 0] = 255
        try:
            im = Image.fromarray(data)
            im = im.convert('L')
            im.save(self.paths.output + name)
        except TypeError:
            data = cv2.convertScaleAbs(data)
            im = Image.fromarray(data)
            im = im.convert('L')
            im.save(self.paths.output + name)

    def stack(self, pixel_array, stack_obj, naming_convention, extension='IMA'):
        """
        Save pixel array as DICOM stack with metadata based on the example stack object
        """
        stack = deepcopy(stack_obj)
        for i in range(stack.info['Stack size']):
            # noinspection PyPep8Naming
            stack.slices[i].PixelData = np.array(pixel_array[i], dtype=stack.slices[i].pixel_array.dtype).tobytes()
            pydicom.filewriter.dcmwrite(self.paths.resulting_stack + naming_convention + str(i + 1) + '.' + extension,
                                        stack.slices[i])

    def contour(self, pixel_array, name, bold=False):
        image = np.zeros((self.stack_info['Resolution']['Y'], self.stack_info['Resolution']['X']))
        for i, j in pixel_array:
            image[i, j] = 1
            if bold:
                image[i + 1, j] = 1
                image[i, j + 1] = 1
                image[i + 1, j + 1] = 1
                image[i - 1, j] = 1
                image[i, j - 1] = 1
                image[i - 1, j - 1] = 1
                image[i + 1, j - 1] = 1
                image[i - 1, j + 1] = 1
                image[i + 2, j] = 1
                image[i, j + 2] = 1
                image[i - 2, j] = 1
                image[i, j - 2] = 1
        self.binary(image, name, True)
        return image
