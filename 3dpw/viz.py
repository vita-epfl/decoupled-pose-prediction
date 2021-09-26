"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""

import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Ax3DPose(object):
    def __init__(self, ax, window=1.0, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
            ax: 3d axis to plot the 3d pose on
            lcolor: String. Colour for the left part of the body
            rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I =  np.array([0, 0, 2, 1, 3, 7, 8, 7, 6, 6, 7, 9, 8, 10])
        self.J =  np.array([1, 2, 4, 3, 5, 0, 1, 8, 7, 8, 9, 11, 10, 12])

        # Left / right indicator
        self.LR = np.array([0, 1, 1, 2, 2, 1, 2, 0, 1, 2, 1, 1, 2, 2], dtype=np.int8)
        self.color = {0: "k", 1:lcolor, 2:rcolor}

        self.ax = ax
        self.num_person = 2

        vals = np.zeros((self.num_person, 28, 3))

        # Make connection matrix
        self.plots = []
        for p in range(self.num_person):
            for i in np.arange( len(self.I) ):
                x = np.array( [vals[p, self.I[i], 0], vals[p, self.J[i], 0]] )
                y = np.array( [vals[p, self.I[i], 1], vals[p, self.J[i], 1]] )
                z = np.array( [vals[p, self.I[i], 2], vals[p, self.J[i], 2]] )
                color = 'k'
                self.plots.append(self.ax.plot(x, y, z, lw=2, c=self.color[self.LR[i]]))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.view_init(15, -65)

        self.window = window
        self.center()

        self.ax.set_xlim3d([-self.window+self.xc, self.window+self.xc])
        self.ax.set_ylim3d([-self.window-self.zc, self.window-self.zc])
        self.ax.set_zlim3d([-0.2 + self.yc, self.window*2 + 0.2 + self.yc])

    def center(self, xc=0.0, yc=0.0, zc=0.0):
        self.xc = xc
        self.yc = yc
        self.zc = zc    
        # print('Centered at: {:.2f}, {:.2f}, {:.2f}'.format(xc, yc, zc))

    def update(self, poses):
        """
        Update the plotted 3d pose.

        Args
          poses: [2, 13, 3] np array. The pose to plot.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert poses.shape[0] == self.num_person, "data should be about 2 persons, it has %d instead" % poses.shape[0]

        for p in range(poses.shape[0]):
            for i in np.arange( len(self.I) ):
                x = np.array( [poses[p, self.I[i], 0], poses[p, self.J[i], 0]] )
                y = np.array( [poses[p, self.I[i], 1], poses[p, self.J[i], 1]] )
                z = np.array( [poses[p, self.I[i], 2], poses[p, self.J[i], 2]] )
                self.plots[i+p*len(self.I)][0].set_xdata(x)
                self.plots[i+p*len(self.I)][0].set_ydata(-z)
                self.plots[i+p*len(self.I)][0].set_3d_properties(y)
                self.plots[i+p*len(self.I)][0].set_color(self.color[self.LR[i]])

        self.ax.set_xlim3d([-self.window+self.xc, self.window+self.xc])
        self.ax.set_ylim3d([-self.window-self.zc, self.window-self.zc])
        self.ax.set_zlim3d([-0.2 + self.yc, self.window*2 + 0.2 + self.yc])        
        # self.ax.set_zlim3d([self.yc -0.2 - self.window, self.yc + self.window + 0.2])
        self.ax.set_box_aspect((1, 1, 1))


class Seq3DPose(object):
    """
    Create a 3d pose squence visualizer that can be updated with new frames.
    """

    def __init__(self, window=1.0):
        self.fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        #ax.set_axis_off() 
        self.ob = Ax3DPose(ax, window)

    def center(self, xc, yc, zc):
        self.ob.center(xc, yc, zc)

    def view(self, seq, name=None, prefix=None):
        for frame in range(seq.shape[1]):
            poses = seq[:, frame]
            self.ob.update(poses)
            self.fig.show()
            self.ob.ax.set_title('{} #{:d}/{:d}'.format(name, frame+1, seq.shape[1]))
            self.fig.canvas.draw() #BEHNAM
            if prefix:
                self.fig.savefig(prefix + '_' + name + '_{:02d}.png'.format(frame), bbox_inches='tight')
            else:
                self.fig.pause(0.1)

class CameraPose:
    def __init__(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(projection='3d')
        #self.ax.set_aspect("auto")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.window = 0.1

    def center(self, centroid):
        self.xc = centroid[0]
        self.yc = centroid[1]
        self.zc = centroid[2]

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

        self.ax.set_xlim([self.xc-self.window, self.xc+self.window])
        self.ax.set_ylim([self.yc-self.window, self.yc+self.window])
        self.ax.set_zlim([self.zc-self.window, self.zc+self.window])
        self.ax.set_box_aspect((1, 1, 1))

    # def customize_legend(self, list_label):
    #     list_handle = []
    #     for idx, label in enumerate(list_label):
    #         color = plt.cm.rainbow(idx / len(list_label))
    #         patch = Patch(color=color, label=label)
    #         list_handle.append(patch)
    #     plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
