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
        instance = self.read_slice(0)
        resolution = {'X': instance.Columns,
                      'Y': instance.Rows}
        info = {'Stack size': len(self.names),
                'Resolution': resolution,
                'Pixel spacing': instance.PixelSpacing[:],
                'Slice thickness': instance.SliceThickness}
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

    def load(self):
        """Read, sort and collect slices into a single list"""
        slices = []
        for i in range(self.info['Stack size']):
            slices.append(self.read_slice(i))
        self.slices = self.merge_sort(slices)

    def merge_sort(self, m):
        """Recursive merge sort algorithm"""
        if len(m) <= 1:
            return m
        # Split list in half
        middle = len(m) // 2
        left = m[:middle]
        right = m[middle:]

        left = self.merge_sort(left)
        right = self.merge_sort(right)
        return list(self.merge(left, right))

    @staticmethod
    def merge(left, right):
        result = []
        left_idx, right_idx = 0, 0
        while left_idx < len(left) and right_idx < len(right):
            # change the direction of this comparison to change the direction of the sort
            if left[left_idx].SliceLocation <= right[right_idx].SliceLocation:
                result.append(left[left_idx])
                left_idx += 1
            else:
                result.append(right[right_idx])
                right_idx += 1

        if left:
            result.extend(left[left_idx:])
        if right:
            result.extend(right[right_idx:])
        return result


