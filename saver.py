# TODO: finish data saver class. Must be capable of saving pictures of different kinds.
#  Will be initialized from DoubleStack class.
class Saver:
    def save_binary_as_image(self, binary_pixel_array, name, increase_contrast=True):
        """Saves data array as a binary image"""
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
