from scipy.interpolate import interp1d
import numpy as np
import meshio
import cv2
import os


class Discretizer:
    def __init__(self, info, core, paths, save):
        self.paths = paths
        self.core = core
        self.geometry_name = 'geometry.geo'
        self.contour = []
        self.mesh_contour = []
        self.stack_info = info
        self.save = save
        self.geo = None

    def evaluate_geometry(self, elem_side_length):
        """
        Evaluate required geometry parameters for meshing
        """
        im = np.array(self.core, dtype=np.uint8)
        contour, _ = cv2.findContours(image=im,
                                      mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
        for i in range(len(contour[0])):
            self.contour.append(contour[0][i].ravel())
        self.save.contour(self.contour, 'contour.bmp')
        self.evaluate_mesh_contour(elem_side_length)
        self.save.contour(self.mesh_contour, 'mesh_contour.bmp', True)
        self.write_geometry()

    def evaluate_mesh_contour(self, elem_side_length):
        """
        Evaluate mesh contour according to element side length at the core boundary
        """
        # Convert form mm to pixels
        elem_side_length = round(elem_side_length / self.stack_info['Pixel spacing'][0])
        print('Nearest possible side length of the element is %f' %
              (elem_side_length * self.stack_info['Pixel spacing'][0]))

        # Collect x and y coordinates of the contour points in separate arrays
        x = np.append(np.array(self.contour)[:, 1], self.contour[0][1])
        y = np.append(np.array(self.contour)[:, 0], self.contour[0][0])

        # Linear length on the line
        distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))

        # Scale to the range [0, 1]
        distance_scaled = distance / distance[-1]
        elem_side_length_scaled = elem_side_length / distance[-1]

        # Interpolate x and y coordinates
        fx, fy = interp1d(distance_scaled, x), interp1d(distance_scaled, y)
        alpha = np.linspace(0, 1, 1 / elem_side_length_scaled)
        x_new, y_new = fx(alpha), fy(alpha)

        # Round resulting coordinates to pixels
        x_new, y_new = np.around(x_new), np.around(y_new)

        # Reassemble contour
        coord = np.vstack((y_new, x_new)).T
        for i in range(len(x_new)):
            self.mesh_contour.append(coord[i].astype(int))

    # TODO: adjust mesh coordinates conversion to meters from pixels;
    #   check if mesh contour is correct in regard to units
    def mesh(self, name, geo_name=''):
        """
        Launch Gmsh and discretize evaluated geometry
        """
        if not geo_name:
            geo_name = self.geometry_name
        os.system('gmsh %s -3 -o %s' % (self.paths.output + geo_name, self.paths.output + name))
        mesh = meshio.read(self.paths.output + name)
        mesh.points[:, 1:] = np.around(mesh.points[:, 1:])
        meshio.write(self.paths.output + name, mesh)

    def write_geometry(self):
        """
        Write geometry file for Gmsh
        """
        script = '//+\nSetFactory("OpenCASCADE");\nlc = 0;\n'
        for k in range(len(self.mesh_contour)):
            i, j = self.mesh_contour[k]
            script += 'Point(%d) = {0, %d, %d, lc};\n' % (k + 1, j, i)
        for k in range(len(self.mesh_contour) - 2):
            script += 'Line(%d) = {%d, %d};\n' % (k + 1, k + 1, k + 2)
        script += 'Line(%d) = {%d, %d};\n' % (len(self.mesh_contour) - 1, len(self.mesh_contour) - 1, 1)
        script += 'Line Loop(%d) = {' % (len(self.mesh_contour))
        for k in range(len(self.mesh_contour) - 2):
            script += '%d, ' % (k + 1)
        script += '%d};\n' % (len(self.mesh_contour) - 1)
        script += 'Plane Surface(%d) = {%d};\n' % (len(self.mesh_contour) + 1, len(self.mesh_contour))
        script += '\nExtrude{%f, 0, 0}\n' % \
                  (self.stack_info['Slice thickness'] / self.stack_info['Pixel spacing'][0] *
                   (self.stack_info['Stack size'] - 1))
        script += '{\nSurface{%d}; Layers{%d}; Recombine;\n}' % (len(self.mesh_contour) + 1,
                                                                 self.stack_info['Stack size'] - 1)
        self.geo = script
        script += '\n\nPhysical Volume("Inner Volume") = {1};\n'
        file = open(self.paths.output + self.geometry_name, "w+")
        file.write(script)
        file.close()

    def get_fiji_mesh(self):
        """
        Get 2D slice from 3D mesh for later use in Fiji macro
        """
        script = self.geo
        script += '\n\nPhysical Surface("Top") = {%d};\n' % (len(self.mesh_contour) + 1)
        file = open(self.paths.output + 'fiji.geo', "w+")
        file.write(script)
        file.close()
        self.mesh('fiji.msh', 'fiji.geo')
        return self.prepare_mesh('fiji.msh')

    def prepare_mesh(self, name):
        mesh = meshio.read(self.paths.output + name)
        mesh.points = np.around(mesh.points)
        return mesh

    def evaluate_physical_mesh(self, name):
        mesh = meshio.read(self.paths.output + name)
        mesh.points[:, 0] = mesh.points[:, 0] * self.stack_info['Pixel spacing'][0] / 1000
        mesh.points[:, 1] = mesh.points[:, 1] * self.stack_info['Pixel spacing'][0] / 1000
        mesh.points[:, 2] = mesh.points[:, 2] * self.stack_info['Pixel spacing'][0] / 1000
        meshio.write(self.paths.output + 'physical_' + name, mesh)
