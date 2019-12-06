from PIL import Image
from copy import deepcopy
import numpy as np
import pydicom
import cv2


# TODO: finish data saver class. Must be capable of saving pictures of different kinds.
#  Will be initialized from DoubleStack class.
class Saver:
    def __init__(self, output_dir, stack_output_dir):
        self.output = output_dir
        self.stack_output = stack_output_dir

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
            im.save(self.output + name)
        except TypeError:
            data = cv2.convertScaleAbs(data)
            im = Image.fromarray(data)
            im = im.convert('L')
            im.save(self.output + name)

    def stack(self, pixel_array, stack_obj, naming_convention, extension='IMA'):
        """
        Save pixel array as DICOM stack with metadata based on the example stack object
        """
        stack = deepcopy(stack_obj)
        for i in range(stack.info['Stack size']):
            # noinspection PyPep8Naming
            stack.slices[i].PixelData = np.array(pixel_array[i], dtype=stack.slices[i].pixel_array.dtype).tobytes()
            pydicom.filewriter.dcmwrite(self.stack_output + naming_convention + str(i + 1) + '.' + extension,
                                        stack.slices[i])

    # TODO: finish method
    def measurements(self):
        """
        Save measurements as file
        """
        pass
