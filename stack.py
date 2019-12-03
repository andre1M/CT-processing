import pydicom
import re
import os


class Stack:
    def __init__(self, path, extension='IMA'):
        self.path = path
        self.extension = '.*' + extension
        self.names = self.read_names()
        self.info = self.get_info()
        self.slices = None

    def get_info(self):
        """Determine stack crucial information"""
        size = len(self.names)
        instance = self.read_slice(0)
        resolution = {'X': instance.Columns,
                      'Y': instance.Rows}
        pixel_spacing = instance.PixelSpacing[0]
        info = {'Stack size': size,
                'Resolution': resolution,
                'Pixel Spacing': pixel_spacing}
        return info

    def read_names(self):
        """Scan folder with CT and get slice names"""
        all_names = os.listdir(self.path)
        extension = re.compile(self.extension)
        image_names = list(filter(extension.match, all_names))
        return image_names

    def read_slice(self, index):
        """Read a single slice from stack"""
        slice_data = pydicom.dcmread(self.path + self.names[index])
        return slice_data

    # TODO: finish 'load' function. Sorting must be perform according to slice metadata and not by name
    def load(self):
        """Read, sort and collect slices into a single list"""
        self.slices = None
        pass
