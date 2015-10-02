__author__ = 'Chongxi'

import numpy as np
from skimage.io import *
import skimage.transform as transform
from vispy import plot as vp
from vispy import scene

img = MultiImage('./data/data_09_25_2015/cell2/C2_3D.tiff')
vol_data = img[0]
print('Image Loaded, ', vol_data.shape)

mpp = 0.341797
zpp = 0.59
ratio = zpp/mpp
print('Transforming Matrix... 3D resolution equalization')
vol_data = transform.resize(vol_data,(vol_data.shape[0],int(vol_data.shape[1]/ratio),int(vol_data.shape[1]/ratio)), preserve_range=True)
# vol_data = np.load('/Users/Chongxi/PycharmProjects/Signal Processing/tritrode_CAM/data/data_09_25_2015/cell2/c2_3d_2.npy')

print('Rendering with OpenGL')
fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)
vol_data = np.flipud(np.rollaxis(vol_data, 1))
print vol_data.shape

clim = [300, 1000]
fig[0,0].volume(vol_data, clim=clim, method='additive', cmap='grays')
fig[0,0].view.camera = scene.TurntableCamera()

if __name__ == '__main__':
    fig.show(run=True)
