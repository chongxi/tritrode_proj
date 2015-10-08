from vispy.scene import SceneCanvas
from vispy import app, scene
from vispy.io import load_data_file, read_png
from vispy.geometry.generation import create_sphere
from skimage.io import imread


import numpy as np
from numpy.linalg.linalg import norm
from scipy.signal import filtfilt, resample
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from process_lib import butter_bandpass, spkdet, get_spk, marray

##################### link x-axis in two views #####################
from vispy.scene import BaseCamera
from vispy.geometry import Rect
class XSyncCamera(BaseCamera):
    def set_state(self, state=None, **kwargs):
        D = state or {}
        if 'rect' not in D:
            return
        for cam in self._linked_cameras:
            r = Rect(D['rect'])
            if cam is self._linked_cameras_no_update:
                continue
            try:
                cam._linked_cameras_no_update = self
                cam_rect = cam.get_state()['rect']
                r.top = cam_rect.top
                r.bottom = cam_rect.bottom
                cam.set_state({'rect':r})
            finally:
                cam._linked_cameras_no_update = None


def link_x(plotwidget1, plotwidget2):
    x_sync_cam = XSyncCamera()
    plotwidget1.camera.link(x_sync_cam)
    plotwidget2.camera.link(x_sync_cam)


##################### add legend to a view #####################

def add_legend(view, label_str, colors):
	from vispy import plot as vp
	labelgrid = view.add_grid(margin=10)
	hspacer = vp.Widget()
	hspacer.stretch = (6, 1)
	labelgrid.add_widget(hspacer, row=0, col=0)

	box = vp.Widget(bgcolor=(1, 1, 1, 0.2), border_color='k')
	labelgrid.add_widget(box, row=0, col=1)

	vspacer = vp.Widget()
	vspacer.stretch = (1, 3)
	labelgrid.add_widget(vspacer, row=1, col=1)
	# print len(label_str)
	labels = [vp.Label(label_str[i], color=colors[i], anchor_x='left')
						for i in range(len(label_str))]
	boxgrid = box.add_grid(bgcolor=(0,0,0,0.7))
	for i, label in enumerate(labels):
	    boxgrid.add_widget(label, row=i, col=0)
	hspacer2 = vp.Widget()
	hspacer2.stretch = (4, 1)
	boxgrid.add_widget(hspacer2, row=0, col=1)
	return labelgrid, box

##################### check mouse event is in view.camera ##################
def is_in_view(event_pos, cam):
	is_in_xrange = \
	event_pos[0] < cam._viewbox.pos[0] + cam._viewbox.size[0] - cam._viewbox.margin \
	and event_pos[0] > cam._viewbox.pos[0] + cam._viewbox.margin
	
	is_in_yrange = \
	event_pos[1] < cam._viewbox.pos[1] + cam._viewbox.size[1] - cam._viewbox.margin \
	and event_pos[1] > cam._viewbox.pos[1] + cam._viewbox.margin

	inview = is_in_xrange and is_in_yrange
	return inview

def get_center_of_view(cam):
	center_x = cam._viewbox.pos[0] + cam._viewbox.size[0]/2
	center_y = cam._viewbox.pos[1] + cam._viewbox.size[1]/2
	c = (center_x, center_y)
	return c

def in_which_view(event_pos, view):
	for v in view:
		is_in_xrange = \
					event_pos[0] < v.pos[0] + v.size[0] - v.margin \
					and event_pos[0] > v.pos[0] + v.margin
		is_in_yrange = \
					event_pos[1] < v.pos[1] + v.size[1] - v.margin \
					and event_pos[1] > v.pos[1] + v.margin
		inview = is_in_xrange and is_in_yrange
		if inview:
			return v
			break
		else:

			continue
	return None


##################### 2d mouse event to 3d coordinate #####################
def pos2d_to_pos3d(pos, cam):
	"""Convert mouse event pos:(x, y) into x, y, z translations"""
	"""dist is the distance between (x,y) and (cx, cy) of cam"""
	center = get_center_of_view(cam)
	dist = pos - center
	dist[1] *= -1
	rae = np.array([cam.azimuth, cam.elevation]) * np.pi / 180
	saz, sel = np.sin(rae)
	caz, cel = np.cos(rae)
	dx = (+ dist[0] * (1 * caz)
	      + dist[1] * (- 1 * sel * saz))
	dy = (+ dist[0] * (1 * saz)
	      + dist[1] * (+ 1 * sel * caz))
	dz = (+ dist[1] * 1 * cel)

	# Black magic part 2: take up-vector and flipping into account
	ff = cam._flip_factors
	up, forward, right = cam._get_dim_vectors()
	dx, dy, dz = right * dx + forward * dy + up * dz
	dx, dy, dz = ff[0] * dx, ff[1] * dy, ff[2] * dz
	return dx, dy, dz

##################### get xlim and ylim from a view #####################
def get_xlim(view):
	_xlim = np.array([0.0,0.0])
	_xlim[0] = view.camera.get_state()['rect']._pos[0]
	_xlim[1] = _xlim[0] + view.camera.get_state()['rect']._size[0]
	return _xlim

def get_ylim(view):
	_ylim = np.array([-10.0,-10.0])
	_ylim[0] = view.camera.get_state()['rect']._pos[1]
	_ylim[1] = _ylim[0] + view.camera.get_state()['rect']._size[1]
	return _ylim

################## scalar field generator ##################
## Define a scalar field from which we will generate an isosurface
def psi(i, j, k, center=(128, 128, 128)):
    x = i-center[0]
    y = j-center[1]
    z = k-center[2]
    r = (x**2 + y**2 + z**2)
    field = np.exp(-r/(7**2))
    return field

# np.fromfunction:
# The resulting data array has a value fn(x, y, z) at coordinate (x, y, z).
# Bolished because now I use a faster method by isolines animation
# sensor_pos = np.abs(np.fromfunction(psi, (256, 256, 256)))

##################### color table ######################
#49cbd3

_colors = [ '#81d745',
			'#d74e40',
			'#49cbd3' ]
# 469b55
# 973533
# 6d5ba2

##################### Set Canvas and Grid #####################

canvas = SceneCanvas(title='Electrogenesis in Extracellular space', 
					 keys='interactive', bgcolor='w', size=(1200,800), 
					 position=(120,40), show=True) #fullscreen=True, always_on_top=True,
grid = canvas.central_widget.add_grid(spacing=0,bgcolor='#2f3234',border_color='k')

##################### Assign view to Grid #####################
alpha = 0.3
# Image view
view1 = grid.add_view(row=2, col=0, bgcolor=(0,0,0,1),
					  border_color=(1,0,0),
					  margin=8)
# volume view
view2 = grid.add_view(row=0, col=0, row_span=2, bgcolor=(0,0,0,1),
					  border_color=(0,1,0), 
					  margin=15)
# Line1 view
view3 = grid.add_view(row=0, col=1, row_span=1, col_span=3, bgcolor=(0,1,0,alpha-0.25),
					  border_color=(0,1,0),
					  margin=25)
# Line2 view
view4 = grid.add_view(row=1, col=1, row_span=1, col_span=3, bgcolor=(0,1,1,alpha-0.25),
					  border_color=(0,1,1), 
					  margin=25)
# Line3 view (aggretate)
view5 = grid.add_view(row=2, col=1, col_span=1, bgcolor=(1,0,1,alpha),
					  border_color=(1,0,1), 
					  margin=10)
# TBD
view6 = grid.add_view(row=2, col=2, col_span=1, bgcolor=(1,1,1,alpha),
					  border_color=(1,1,1),
					  margin=10)
# TBD
view7 = grid.add_view(row=2, col=3, col_span=1, bgcolor=(1,1,0,alpha),
					  border_color=(1,1,0), 
					  margin=10)

view = (view1,view2,view3,view4,view5,view6,view7)
##################### Data and Meta-data #####################
import igor.igorpy as igor
igor.ENCODING = 'UTF-8'
 
###############################################################

# L23 09-28
# recording a circle around cell indicate no radius

exp_path = './data/2015-09-28/'
igo_path = './data/2015-09-28/l23-c1-1.pxp'
img_path = './data/2015-09-28/L23_c1/'
img_3d_name = 'c1_3d.npy'
_clim=(50,200)

###############################################################
# CA1 09-28 
# CA1 big signal at axon
# Dendrite cable 
# s4: 15um=> 0.17
# s3: 15um=> 0.28
# s2: 16um=> 1 (closer to axon)
# s5: 5um => 0.78 (far from axon)
# a8: bouton
# a9->a4 decreasing
# d1<=>s3 cable > soma

# exp_path = './data/2015-09-28/'
# igo_path = './data/2015-09-28/CA1_c1.pxp'
# img_path = './data/2015-09-28/CA1_c1/'
# img_3d_name = 'c1_3d.npy'
# _clim=(100,1000)
###############################################################

# CA1 09-29
# trace dendrite down to 42um: rate: 1
# 1. waveform polarity
# 2. cable dependent
# 3. soma dcrease much faster: d4<=>s4

# exp_path = './data/2015-09-29/'
# igo_path = './data/2015-09-29/CA1c2.pxp'
# img_path = './data/2015-09-29/CA1c2/'
# img_3d_name = 'c1_3d.npy'
# _clim=(100,500)

###############################################################
# giant spikes from other cells
# this cell has perfect intracellular firing but very weak EAP
# CA1 09-30: population imaging
# exp_path = './data/2015-09-30/'
# igo_path = './data/2015-09-30/CA1c2-2.pxp'
# img_path = './data/2015-09-30/CA1c2/'
# img_3d_name = 'c2_3d.npy'
# _clim=(100,500)

###############################################################

# CA1 10-05 
# both axon and dendrite
# s1: 14um no signal. 
# ad6: axon 2 peaks. delay between axon EAP and dendrite EAP
# s2: 500uV from other cell, only show up in right channel, while the second
# channel is just 25um away. where is Volume conduction?
# exp_path = './data/2015-10-05/'
# igo_path = './data/2015-10-05/CA1c1_axondendrite.pxp'
# img_path = './data/2015-10-05/CA1c1/'
# img_3d_name = 'c1_3d.npy'
# _clim=(50,500)

# m_test
# exp_path = './data/2015-10-02/'
# igo_path = './data/2015-10-02/m_test.pxp'
# img_path = './data/2015-10-02/m_test/'
# img_3d_name = 'c1_3d.npy'
# _clim=(50,1500)





global id_legend, _id, t, intra_trace, extra_trace, img_fname
datum = {}
log = {}

print('Experiment path @ %s' % exp_path)

# igor file (electrophysiology)
print('Loading igor file @ %s' % igo_path)
ig = igor.load(igo_path)
_id_igor = dir(ig)

# imaging tiff
import os
print('loading imaging @ %s' % img_path)
for filename in os.listdir(img_path):
    if filename.endswith(".tiff"):
        datum[filename.split('_')[0].lower()] = [filename]

# log and mpp
print('loading logs @ %s' % img_path)
for filename in os.listdir(img_path):
    if filename.endswith(".log"):
        log[filename.split('_')[0].lower()] = [filename]

# generate mpp
import re
for _id in log.keys():
    if _id in _id_igor:
        infile = img_path+log[_id][0]
        with open(infile) as f:
            f = f.readlines()
            for line in f:
                if 'Microns Per Pixel' in line:
                    mpp = re.findall('Microns Per Pixel: ([\d.]+)', line)
                    log[_id].append(mpp)
    else:
        log.pop(_id)

# function to get mpp
def get_mpp(_id):
	mpp = float(log[_id][1][0])
	return mpp

# datum is dictionary contains tiff and igor waveform
print('integrating imaging and electrophysiology')
for _id in datum.keys():
    if _id in _id_igor:
        datum[_id].append(ig[_id])
    else:
        datum.pop(_id)

for keys,values in datum.items():
    print(keys)
    print(values)

# 3d volume data (neuron morphology)
vol = np.load(img_path+img_3d_name)
vol = np.flipud(np.swapaxes(vol, 0, 1))
print('loaded volume shape:' , vol.shape)

global i
i = 0
n_datum = len(datum.keys())
# extract experiment point
def extract_data(i):
	global id_legend, _id, t, intra_trace, extra_trace, img_fname
	if i < n_datum and i>=0:
		_id = datum.keys()[i] # 'a0' or 's1' etc.
		img_fname = img_path+datum[_id][0]
		igor_trace = datum[_id][1]
		extra_id_0 = _id + '1_Ax1_Vm' # Axon
		extra_id_1 = _id + '1_Ch2_Vm' # Dendrite
		intra_id = _id + '1_Ch1_Vm'
		t = igor_trace[intra_id].axis[0] * 1e6
		t = t[::-1]
		intra_trace = igor_trace[intra_id].data * 1e3 # unit(mV)
		extra_trace = np.zeros((len(t),2))
		extra_trace[:,0] = igor_trace[extra_id_0].data * 1e6 # unit(uV) Axon (left)
		extra_trace[:,1] = igor_trace[extra_id_1].data * 1e6 # unit(uV) Dendrite (right)
		##################### down-sampling #####################
		N = 2 # down sampling 2 folds
		intra_trace,t_d = resample(intra_trace, intra_trace.shape[0]/N, t)
		extra_trace,t_d = resample(extra_trace, extra_trace.shape[0]/N, t)
		t = t_d
		print 'estimated fs ',1/(t[2]-t[1])
		###################### filtering process #####################
		fs = np.ceil(1/(t[2]-t[1]))
		print('sampling frequency %f' % fs)
		b, a = butter_bandpass(200,3000,fs,6)
		extra_trace = filtfilt(b, a, extra_trace.T, padlen=150, padtype="even")
		extra_trace = extra_trace.T
		###############################################################
		return _id, t, intra_trace, extra_trace, img_fname
	elif i<0:
		print('out of range, cannot less than 0')
	elif i >= n_datum:
		print('out of range, cannot less than %d' % n_datum)

_id, t, intra_trace, extra_trace, img_fname = extract_data(i)

##################### Put img on one view1 #####################
# measure_line.visible=True
arr = np.array([(0, 100), (100, 0)])
measure_line = scene.visuals.Line(pos=arr, parent=view1.scene, color='g')
measure_line.visible = False

# img_data = read_png(load_data_file('mona_lisa/mona_lisa_sm.png'))
img_data = imread(fname=img_fname)
print('loaded 2P imaging shape', img_data.shape)
image = scene.visuals.Image(img_data, parent=view1.scene, cmap='grays', clim=_clim) #,clim=(100,2000)
view1.camera = scene.PanZoomCamera(aspect=1)
view1.camera.flip = (0,1,0)
view1.camera.set_range()
# view1.camera.zoom(1, view1.camera.center)
# legend for view1: _id (experiment id)
view1_text = scene.Text(parent=view1.scene, color='red')
view1_text.font_size = 14
view1_text.pos = (50,50)
view1_text.text = _id

measure_text = scene.Text(parent=view1.scene, color='g')
measure_text.font_size = 10
measure_text.pos = (0,0)
measure_text.visible = False

@canvas.connect
def on_mouse_move(event):
	if event.press_event is None:
		return

	modifiers = event.modifiers
	pos = event.press_event.pos
	if is_in_view(pos, view1.camera):
		if modifiers is not ():
			if 1 in event.buttons and modifiers[0].name=='Control':
				# Translate: camera._scene_transform.imap(event.pos)
				p1 = np.array(pos)[:2]
				p2 = np.array(event.last_event.pos)[:2]
				p1 = p1 - view1.pos
				p2 = p2 - view1.pos
				# print p1,p2
				p1s = view1.camera._scene_transform.imap(p1)[:2]
				p2s = view1.camera._scene_transform.imap(p2)[:2]
				print p1s, p2s
				pos_ = np.vstack((p2s,p1s))
				# print pos_
				measure_line.set_data(pos=pos_)
				measure_line.visible = True
				d_pixel = norm(pos_[1,:]-pos_[0,:])
				d_um = d_pixel*get_mpp(_id)
				print 'distance =',d_um
				measure_text.visible = True
				measure_text.text = '%.2f um' % d_um
				measure_text.pos = pos_[1,:]
				measure_text.pos[0] -= 10
				event.handled = True

		# arr = np.array([(200, 0), (0, 200)])
		# measure_line.set_data(pos=arr)

##################### Volume to another view2 #####################
# vol = np.load(load_data_file('brain/mri.npz'))['data']
# vol = np.flipud(np.swapaxes(vol, 0, 1))
# print vol.shape

volume = scene.Volume(vol, parent=view2.scene, clim=_clim,
				      emulate_texture=True)
# volume.transform = scene.STTransform(translate=(0, 0, 200))
# scene.visuals.XYZAxis(parent=view2.scene)
view2.camera = scene.TurntableCamera(parent=view2.scene)
view2.camera.set_range()
view2.camera.azimuth = 0
view2.camera.elevation = 0
init_scale_factor = view2.camera._scale_factor

################ Isolines animation added to view2 ################
sensor_pos = (110,90,128)
radius = 2
cols = 10
rows = 10

amination_alpha = 1
iso = scene.Isoline(parent=view2.scene)
iso.set_color((0,1,1,amination_alpha))
iso.transform = scene.transforms.STTransform(translate=sensor_pos)

def sensor_animation(ev):
    global radius, amination_alpha
    radius += 2
    amination_alpha -= 0.4
    if radius > 7:
        radius = 2
        amination_alpha = 1
    mesh = create_sphere(cols, rows, radius=radius)
    vertices = mesh.get_vertices()
    tris = mesh.get_faces()
    nbr_level = 20
    cl = np.linspace(-radius, radius, nbr_level+2)[1:-1]
    iso.set_data(vertices=vertices, tris=tris, data=vertices[:, 2])
    iso.levels=cl
    iso.set_color((0,1,1,amination_alpha))

# set timer2 to animate the position of sensor
timer2 = app.Timer()
timer2.connect(sensor_animation)
timer2.start(0.15)


##################### Line1 add to view3 #####################
x_axis1 = scene.AxisWidget(orientation='bottom', axis_color=(0,1,0),
						   tick_color=(0,1,0), text_color=(0,1,0),
						   font_size=7)
y_axis1 = scene.AxisWidget(orientation='left', axis_color=(0,1,0),
						   tick_color=(0,1,0), text_color=(0,1,0),
						   font_size=7)
grid.add_widget(x_axis1, row=1, col=1, col_span=3)
grid.add_widget(y_axis1, row=0, col=0)
x_axis1.margin=-25
y_axis1.margin=-25

# N = 116000
# pos = np.zeros((N,2), dtype=np.float)
# pos[:,0] = np.linspace(0, 10, N)
# pos[:,1] = np.cos(pos[:,0]) + np.random.randn(len(pos[:,0]))
# special_point = np.where(np.logical_and(0.3812<pos[:,0], pos[:,0]<0.3813))
# pos[special_point,1] = -10
trace1 = np.vstack((t, intra_trace)).T   # Intracellular
line1 = scene.Line(pos=trace1, color=_colors[0], parent=view3.scene)
view3.camera = scene.PanZoomCamera()
view3.camera.set_range()
view3.camera.zoom(1.1, view3.camera.center)
x_axis1.link_view(view3)
y_axis1.link_view(view3)


##################### Line2 add to view4 #####################
x_axis2 = scene.AxisWidget(orientation='bottom', axis_color=(0,1,1),
						   tick_color=(0,1,1), text_color=(0,1,1),
						   font_size=7)
y_axis2 = scene.AxisWidget(orientation='left', axis_color=(0,1,1),
						   tick_color=(0,1,1), text_color=(0,1,1),
						   font_size=7)
grid.add_widget(x_axis2, row=2, col=1, col_span=3)
grid.add_widget(y_axis2, row=1, col=0)
x_axis2.margin=-25
y_axis2.margin=-25

# N = 116000
# pos = np.zeros((N,2), dtype=np.float)
# pos[:,0] = np.linspace(0, 10, N)
# pos[:,1] = 30*np.sin(pos[:,0]) + np.random.randn(len(pos[:,0]))
trace2 = np.vstack((t, extra_trace[:,0])).T   # Axon
line2 = scene.Line(pos=trace2, color=_colors[1], parent=view4.scene)
trace3 = np.vstack((t, extra_trace[:,1])).T   # Dendrite
line3 = scene.Line(pos=trace3, color=_colors[2], parent=view4.scene)
view4.camera = scene.PanZoomCamera()
view4.camera.set_range()
x_axis2.link_view(view4)
y_axis2.link_view(view4)

##################### Link x-axis of Line2 and Line3 #####################
link_x(view3, view4)

# --------------------------------------------------------------------------
##################### Extracellular Processing view4 #####################
def spk_show(waves, color, mode, filename):
	fig = plt.figure()
	plt.plot(waves.T,c=color,lw=4)
	plt.axis('off')
	if mode == 'save':
		plt.savefig(filename, bbox_inches='tight', transparent=True)
		plt.close(fig)
	elif mode == 'show':
		plt.axis('on')
		plt.show()
	# for l in waves:
	# 	pos_ = np.zeros((len(l),2))
	# 	pos_[:,0] = np.arange(len(l))
	# 	pos_[:,1] = l
	# 	line = scene.visuals.Line(pos=pos_, color='g', parent=view.scene)
		# lines.append(line)
	# return lines
# --------------------------------------------------------------------------

##################### Add Legend to view 1,3,4 #######################

# legend for view3,4:   
add_legend(view3, ['1.Intracellular(mV)'], ['g'])
add_legend(view4, ['2.Axon(uV)', '3.Dendrite(uV)'], [(1,0,0),(0,1,1)])


#  --------------------------------------------------------------------------

##################### spike extraction and display #####################
# x_axis3 = scene.AxisWidget(orientation='bottom', axis_color=(0,1,1),
# 						   tick_color=(0,1,1), text_color=(0,1,1),
# 						   font_size=7)
# y_axis3 = scene.AxisWidget(orientation='left', axis_color=(0,1,1),
# 						   tick_color=(0,1,1), text_color=(0,1,1),
# 						   font_size=7)
# # grid.add_widget(x_axis3, row=2, col=1)
# grid.add_widget(y_axis3, row=2, col=0)
# # x_axis3.margin = -25
# y_axis3.margin = -25

####################################################################################
global spk_epoch, spk_wave_intra, spk_wave_extra, extra_thr
global spikes_extracted # indicate spike has been extracted
global spike_detect_rate

spikes_extracted = False
spike_detect_rate = marray([0])
spike_No = marray([0])

def extract_spikes(t, intra_trace, extra_trace):
	print 'extracting spikes...pleas wait'
	spk_epoch, spk_wave_intra, spk_wave_extra = get_spk(t, intra_trace, extra_trace)
	print('%d spikes extracted' % spk_wave_intra.shape[0])
	return spk_epoch, spk_wave_intra, spk_wave_extra

def get_spk_detrate(waves, thr):
	N = waves.shape[0]
	k = 0.0
	for wav in waves:
		if wav.min() < thr:
			k += 1
	return k/N, N

# spk_show(spk_wave_intra, 'g', mode='save', filename='./intra.png')
# intra_spk_img = read_png('./intra.png')

# spk_show(spk_wave_extra[:,:,0], 'r', mode='save', filename='./axon.png')
# extra_spk_img0 = read_png('./axon.png')

# spk_show(spk_wave_extra[:,:,1], (0,1,1), mode='save', filename='./den.png')
# extra_spk_img1 = read_png('./den.png')


# # intra
# image_intra = scene.Image(intra_spk_img, parent=view5.scene)
# view5.camera = scene.PanZoomCamera(aspect=1)
# view5.camera.flip = (0,1,0)
# view5.camera.set_range()

# # axon
# image_extra0 = scene.Image(extra_spk_img0, parent=view6.scene)
# view6.camera = scene.PanZoomCamera(aspect=1)
# view6.camera.flip = (0,1,0)
# view6.camera.set_range()

# # dendrite
# image_extra1 = scene.Image(extra_spk_img1, parent=view7.scene)
# view7.camera = scene.PanZoomCamera(aspect=1)
# view7.camera.flip = (0,1,0)
# view7.camera.set_range()
####################################################################################

# view5.camera.zoom(1.2, view5.camera.center)
# view5.camera.interactive=False

# x_axis3.link_view(view5)
# y_axis3.link_view(view5)

# --------------------------------------------------------------------------

##################### timer1 to update: play_trace() #####################
step = 0.02
play_status = False
end_of_trace = t[-1]
view_to_play = view3

def play_trace(ev): 
	global step, play_status, end_of_trace
	# xlim = xlim + step
	cam_rect = view_to_play.camera.get_state()['rect']
	cam_rect._pos = (cam_rect._pos[0]+step, cam_rect._pos[1]) 
	# view_to_play.camera.set_range(x=(xlim[0],xlim[1]),y=(ylim[0],ylim[1]))
	view_to_play.camera.set_state({'rect':cam_rect})
	if get_xlim(view_to_play)[1] > end_of_trace:
		play_stop(timer1)

def play_stop(timer):
	global play_status
	timer.stop()
	play_status = False
	print('playing status: %s' % play_status)

def play_start(timer): 
	global play_status
	timer.start(0.1)
	play_status = True
	print('playing status: %s' % play_status)

timer1 = app.Timer()
timer1.connect(play_trace)

##################### Key press event #####################
@canvas.connect
def on_key_press(event):
	global play_status, step, view_to_play, line1
	global spk_wave_intra, spk_wave_extra, extra_thr, spikes_extracted
	if event.text == ' ' and play_status == False:
		xlim = get_xlim(view_to_play)
		step  = (xlim[1]-xlim[0])/20.0
		play_start(timer1)

	elif event.text == ' ' and play_status == True:
		play_stop(timer1)

	elif event.text == 'l':
		print 'l'
		arr = np.array([(200, 0), (0, 200)])
		measure_line.set_data(pos=arr)

	elif event.text in ['+','=']:
		xlim = get_xlim(view_to_play) 
		step += (xlim[1]-xlim[0])/60.0
		print('playing speed(step) is set to %f' % step)

	elif event.text in ['-','_']:
		xlim = get_xlim(view_to_play)
		step -= (xlim[1]-xlim[0])/60.0
		print('playing speed(step) is set to %f' % step)

	elif event.text == 'x':
		xlim = get_xlim(view_to_play)
		print xlim
		_xrange = xlim[1]-xlim[0]
		print('xrange: %f' % _xrange)
		# print get_xlim(view_to_play)
	elif event.text == 'y':
		ylim = get_ylim(view_to_play)
		print ylim
		# print get_ylim(view_to_play)
	elif event.text == '1':
		line1.visible = False if line1.visible else True
	elif event.text == '2':
		line2.visible = False if line2.visible else True
	elif event.text == '3':
		line3.visible = False if line3.visible else True
	elif event.text == 't':
		fig = plt.figure()
		x = np.linspace(0, 15, 31)
		data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
		sns.tsplot(data=data, err_style="boot_traces", n_boot=500)
		plt.show()
	elif event.text == 'z':
		view2.camera.orbit(azim=1,elev=0)
		print('(azimuth=%f), (elevation=%f), (roll=%f)' 
			% (view2.camera.azimuth, view2.camera.elevation, view2.camera.roll))
	elif event.text == 'Z':
		view2.camera.orbit(azim=-1,elev=0)
		print('(azimuth=%f), (elevation=%f), (roll=%f)' 
			% (view2.camera.azimuth, view2.camera.elevation, view2.camera.roll))
	elif event.text == 'f':
		spk_epoch, spk_wave_intra, spk_wave_extra = \
		extract_spikes(t, intra_trace, extra_trace)
		extra_thr = 3.6 * np.median(abs(extra_trace), axis=0)/0.6745
		spike_detect_rate[0], spike_No[0] = get_spk_detrate(spk_wave_extra[:,:,0], -extra_thr[0])
		spike_detect_rate[1], spike_No[1] = get_spk_detrate(spk_wave_extra[:,:,1], -extra_thr[1])
		spikes_extracted = True
	elif event.text =='v':
		view1.interactive = False if view1.interactive else True

##################### mouse press event #####################
def reset():
	view1.camera.set_range()
	# view1.camera.zoom(1, view1.camera.center)
	view2.camera.set_range()
	view2.camera.azimuth = 0
	view2.camera.elevation = 0
	view3.camera.set_range()
	view3.camera.zoom(1.1, view3.camera.center)

@canvas.connect
def on_mouse_double_click(event):
	global spk_wave_intra, spk_wave_extra, extra_thr, spikes_extracted
	# print event.pos
	if is_in_view(event.pos, view3.camera):
		if spikes_extracted == False:
			spk_epoch, spk_wave_intra, spk_wave_extra = \
			extract_spikes(t, intra_trace, extra_trace)
			extra_thr = 3.6 * np.median(abs(extra_trace), axis=0)/0.6745
			spike_detect_rate[0], spike_No[0] = get_spk_detrate(spk_wave_extra[:,:,0], -extra_thr[0])
			spike_detect_rate[1], spike_No[1] = get_spk_detrate(spk_wave_extra[:,:,1], -extra_thr[1])
			spikes_extracted = True
		else:
			# intra
			plt.figure(facecolor='white', figsize=(10,10))
			ax1 = plt.subplot(311)
			ax1.plot(spk_wave_intra.T, c=_colors[0])
			peak_idx = []
			for l in spk_wave_intra:
				peak_idx.append(l.argmax())
				ax1.plot(l.argmax(), l.max(), 'ro', ms=5)
			p = np.mean(peak_idx)
			str_p = '%.2f' % p
			plt.axvline(p,lw=2,ls='-.',c='m')
			plt.text(p,0,str_p,rotation=90)
			# axon
			ax2 = plt.subplot(312)
			ax2.plot(spk_wave_extra[:,:,0].T, c=_colors[1])
			peak_idx = []
			for l in spk_wave_extra[:,:,0]:
				peak_idx.append(l.argmin())
				ax2.plot(l.argmin(), l.min(), 'go', ms=5)
			p = np.mean(peak_idx)
			str_p = '%.2f' % p
			plt.axvline(p,lw=2,ls='-.',c='m')
			plt.text(p,0,str_p,rotation=90)
			# den
			ax3 = plt.subplot(313)
			ax3.plot(spk_wave_extra[:,:,1].T, c=_colors[2])
			peak_idx = []
			for l in spk_wave_extra[:,:,1]:
				peak_idx.append(l.argmin())
				ax3.plot(l.argmin(), l.min(), 'ro', ms=5)
			p = np.mean(peak_idx)
			str_p = '%.2f' % p
			plt.axvline(p,lw=2,ls='-.',c='m')
			plt.text(p,0,str_p,rotation=90)
			# show()
			plt.show()
	elif is_in_view(event.pos, view4.camera):
		if spikes_extracted == False:
			spk_epoch, spk_wave_intra, spk_wave_extra = \
			extract_spikes(t, intra_trace, extra_trace)
			extra_thr = 3.6 * np.median(abs(extra_trace), axis=0)/0.6745
			spike_detect_rate[0], spike_No[0] = get_spk_detrate(spk_wave_extra[:,:,0], -extra_thr[0])
			spike_detect_rate[1], spike_No[1] = get_spk_detrate(spk_wave_extra[:,:,1], -extra_thr[1])
			spikes_extracted = True
		else:
			plt.figure(facecolor='white')
			plt.plot(spk_wave_extra[:,:,0].T, c=_colors[1])
			plt.plot(spk_wave_extra[:,:,1].T, c=_colors[2])
			# sns.tsplot(data=spk_wave_extra[:,:,0], err_style="unit_traces", c=_colors[1])
			# sns.tsplot(data=spk_wave_extra[:,:,1], err_style="unit_traces", c=_colors[2])
			plt.show()
	if is_in_view(event.pos, view5.camera):
		plt.figure(facecolor='white')
		plt.plot(spk_wave_intra.T, c=_colors[0])
		for l in spk_wave_intra:
			plt.plot(l.argmax(), l.max(), 'ro', ms=5)
			p = l.argmax()
		plt.axvline(p,lw=2,ls='-.',c='m')
		plt.show()
	elif is_in_view(event.pos, view6.camera):
		s = 'rate=%f, number=%d' % (spike_detect_rate[0], spike_No[0])
		plt.figure(facecolor='white')
		# plt.plot(spk_wave_extra[:,:,0].T, c = 'r')
		sns.despine(left=True, bottom=True, right=False, top=False)
		sns.tsplot(data=spk_wave_extra[:,:,0], value="Voltage(uV)", 
					err_style="unit_traces", c=_colors[1], legend=True,
					condition=s)
		plt.axhline(-extra_thr[0], c='m', ls='--', lw=2)
		print('extra_thr=%f' % -extra_thr[0])
		plt.show()
	elif is_in_view(event.pos, view7.camera):
		s = 'rate=%f, number=%d' % (spike_detect_rate[1], spike_No[1])
		plt.figure(facecolor='white')
		# plt.plot(spk_wave_extra[:,:,1].T, c = (0,1,1))
		sns.despine(left=True, bottom=True, right=False, top=False)
		sns.tsplot(data=spk_wave_extra[:,:,1], value="Voltage(uV)", 
					err_style="unit_traces", c=_colors[2], legend=True,
			 		condition=s)
		plt.axhline(-extra_thr[1], c='m', ls='--', lw=2)
		print('extra_thr=%f' % -extra_thr[1])
		plt.show()
	else:
		reset()

@canvas.connect
def on_mouse_press(event):
	modifiers = event.modifiers
	button = event.button
	pos = event.pos
	if modifiers is not ():
		mod = [key.name for key in event.modifiers]
		if mod == ['Control'] and is_in_view(pos, view2.camera) == True:
			# Black magic part 1: turn 2D into 3D translations
			dx,dy,dz = pos2d_to_pos3d(pos,view2.camera)
			# Black magic part 2: scale for mapping exact mouse event pos
			c = view2.camera.center
			scale = 1.50 * 1.48 * view2.camera._scale_factor / init_scale_factor
			x,y,z = c[0] + scale*dx, c[1] + scale*dy, c[2] + scale*dz
			iso.transform = scene.transforms.STTransform(translate=(x,y,z))


##################### Mouse move event, current view #####################

current_view = view1
@canvas.connect
def on_mouse_move(event):
	global current_view
	current_view = in_which_view(event.pos, view)
	# print str(current_view)
	for _v in view:
		if _v is current_view:
			_v._border_width = 1
			_v._update_child_widgets()
			_v._update_line()
			_v.update()
			_v.events.resize()
		else:
			_v._border_width = 0
			_v._update_child_widgets()
			_v._update_line()
			_v.update()
			_v.events.resize()			


##################### Mouse wheel, changing data #####################
def update(i):
	##################### update section 1 #####################
	global id_legend, _id, t, intra_trace, extra_trace, img_fname

	_id, t, intra_trace, extra_trace, img_fname = extract_data(i)
	# update line1
	trace1 = np.vstack((t, intra_trace)).T
	line1.set_data(pos=trace1)
	view3.camera.set_range((0,trace1[-1,0]))
	view3.camera.zoom(1.1, view3.camera.center)
	# update line2 and line3 
	trace2 = np.vstack((t, extra_trace[:,0])).T
	line2.set_data(pos=trace2)
	trace3 = np.vstack((t, extra_trace[:,1])).T
	line3.set_data(pos=trace3)
	# view4.camera.set_range((0,trace1[-1,0]))
	# view4.camera.zoom(1.1, view3.camera.center)
	# update image
	img_data = imread(fname=img_fname)
	image.set_data(img_data)
	# view1.camera = scene.PanZoomCamera(aspect=1)
	# view1.camera.flip = (0,1,0)
	view1.camera.set_range()
	# view1.camera.zoom(1, view1.camera.center)
	view1_text.text = _id

	measure_line.visible = False
	measure_text.visible = False

	##################### update section 2 ####################
	global spikes_extracted
	spikes_extracted = False
	# # spikes etc.... to do:
	# global spk_epoch, spk_wave_intra, spk_wave_extra
	# print 'extracting spikes...pleas wait'
	# spk_epoch, spk_wave_intra, spk_wave_extra = get_spk(t, intra_trace, extra_trace)
	# print('%d spikes extracted' % spk_wave_intra.shape[0])

	# spk_show(spk_wave_intra, 'g', mode='save', filename='./intra.png')
	# intra_spk_img = read_png('./intra.png')
	# spk_show(spk_wave_extra[:,:,0], 'r', mode='save', filename='./axon.png')
	# extra_spk_img0 = read_png('./axon.png')
	# spk_show(spk_wave_extra[:,:,1], (0,1,1), mode='save', filename='./den.png')
	# extra_spk_img1 = read_png('./den.png')

	# # intra
	# image_intra.set_data(intra_spk_img)
	# view5.camera.set_range()

	# # axon
	# image_extra0.set_data(extra_spk_img0)
	# view6.camera.set_range()

	# # dendrite
	# image_extra1.set_data(extra_spk_img1)
	# view7.camera.set_range()
	###############################################################

@canvas.connect
def on_mouse_wheel(event):
	global i
	pos = event.pos
	wheel = event.delta[1]
	if is_in_view(pos, view1.camera):
		view1.camera.interactive = False
		if wheel == -1:  #'up'
			i -= 1
			if i<0:
				i=0
			else:
				update(i)
		elif wheel == 1: #'down'
			i += 1
			if i==n_datum:
				i=n_datum-1
			else:
				update(i)


###############################################################
if __name__ == '__main__':
	canvas.app.run()
