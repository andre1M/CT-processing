import numpy as np
import cv2


# TODO: finish class
class Discretizer:
    def __init__(self, output_dir, core):
        self.output = output_dir
        self.core = core
        self.contour = []
        self.mesh_contour = []

    def evaluate_geometry(self, elem_side_length):
        im = np.array(self.core, dtype=np.uint8)
        contour, _ = cv2.findContours(image=im,
                                      mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
        for i in range(len(contour[0])):
            self.contour.append(contour[0][i].ravel())
        self.evaluate_mesh_contour(elem_side_length)

    def evaluate_mesh_contour(self, elem_side_length):
        # for i in range(0, len(self.contour) - elem_side_length, elem_side_length):
        #     self.mesh_contour.append(self.contour[i])
        pass

    def mesh(self):
        pass





    def get_mesh_contour(contour, grid_side_length):
        mesh_contour = []
        for i in range(0, len(contour) - grid_side_length, grid_side_length):
            mesh_contour.append([contour[i][1], contour[i][0]])
        return mesh_contour

    def write_gmsh_geometry(self, contour, name):
        """Generates a set of commands for Gmsh to create an object for further meshing"""
        string = '//+\nSetFactory("OpenCASCADE");\nlc = 0;\n'
        for k in range(len(contour)):
            i, j = contour[k]
            string += 'Point(%d) = {%d, %d, 0, lc};\n' % (k + 1, j, i)
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