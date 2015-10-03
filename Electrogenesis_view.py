from vispy.scene import SceneCanvas
from vispy import app, scene
from vispy.io import load_data_file, read_png
from vispy.geometry.generation import create_sphere
import numpy as np


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

def add_legend(view, label_str, color):
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

	labels = [vp.Label(label_str, color=color, anchor_x='left')]
	boxgrid = box.add_grid()
	for i, label in enumerate(labels):
	    boxgrid.add_widget(label, row=i, col=0)
	hspacer2 = vp.Widget()
	hspacer2.stretch = (4, 1)
	boxgrid.add_widget(hspacer2, row=0, col=1)

##################### check mouse event is in view.camera ##################
def is_in_view(event_pos, cam):
	is_in_xrange = event_pos[0] < cam._viewbox.pos[0] + cam._viewbox.size[0] 
	is_in_yrange = event_pos[1] < cam._viewbox.pos[1] + cam._viewbox.size[1]
	return is_in_xrange and is_in_yrange

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
sensor_pos = np.abs(np.fromfunction(psi, (256, 256, 256)))

##################### color table ######################

colors = [(1,0,0),
          (0,0,0),
          (0,1,0),
          (0,1,1), ]

##################### Set Canvas and Grid #####################

canvas = SceneCanvas(title='Electrogenesis in Extracellular space', 
					 keys='interactive', bgcolor='w', size=(1200,800), 
					 always_on_top=True, position=(120,40), show=True) #fullscreen=True
grid = canvas.central_widget.add_grid(spacing=0,bgcolor='#2f3234',border_color='k')

##################### Assign view to Grid #####################
alpha = 0.3
# Image view
view1 = grid.add_view(row=2, col=0, bgcolor=(1,0,0,alpha),
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

##################### Put img on one view1 #####################
img_data = read_png(load_data_file('mona_lisa/mona_lisa_sm.png'))
print img_data.shape
image = scene.Image(img_data, parent=view1.scene)
view1.camera = scene.PanZoomCamera(aspect=1)
view1.camera.flip = (0,1,0)
view1.camera.set_range() # important

##################### Volume to another view2 #####################
vol = np.load(load_data_file('brain/mri.npz'))['data']
vol = np.flipud(np.swapaxes(vol, 0, 1))
print vol.shape
volume = scene.Volume(vol, parent=view2.scene, threshold=0.225,
                               emulate_texture=True)
# volume.transform = scene.STTransform(translate=(0, 0, 200))
# scene.visuals.XYZAxis(parent=view2.scene)
view2.camera = scene.TurntableCamera(parent=view2.scene)
view2.camera.set_range()
view2.camera.elevation = 0
view2.camera.azimuth = 0
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

N = 116000
pos = np.zeros((N,2), dtype=np.float)
pos[:,0] = np.linspace(0, 10, N)
pos[:,1] = np.cos(pos[:,0]) + np.random.randn(len(pos[:,0]))
special_point = np.where(np.logical_and(0.3812<pos[:,0], pos[:,0]<0.3813))
pos[special_point,1] = -10
line1 = scene.Line(pos=pos, color=(0,1,0), parent=view3.scene)
view3.camera = scene.PanZoomCamera()
view3.camera.set_range()
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

N = 116000
pos = np.zeros((N,2), dtype=np.float)
pos[:,0] = np.linspace(0, 10, N)
pos[:,1] = 30*np.sin(pos[:,0]) + np.random.randn(len(pos[:,0]))
line2 = scene.Line(pos=pos, color=(0,1,1), parent=view4.scene)
view4.camera = scene.PanZoomCamera()
view4.camera.set_range()
x_axis2.link_view(view4)
y_axis2.link_view(view4)

##################### Link x-axis of Line2 and Line3 #####################
link_x(view3, view4)


##################### Add Legend to view3 and view4 #######################
add_legend(view3, 'Intracellular(mV)', 'w')
add_legend(view4, 'Extracellular(uV)', 'w')


##################### Timer to update #####################
step = 0.02
play_status = False
end_of_trace = pos[-1,0]
view_to_play = view3

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

def play_trace(ev): 
	global step, play_status, end_of_trace
	# xlim = xlim + step
	cam_rect = view_to_play.camera.get_state()['rect']
	cam_rect._pos = (cam_rect._pos[0]+step, cam_rect._pos[1]) 
	# view_to_play.camera.set_range(x=(xlim[0],xlim[1]),y=(ylim[0],ylim[1]))
	view_to_play.camera.set_state({'rect':cam_rect})
	if get_xlim(view_to_play)[1] > end_of_trace:
		play_stop(timer1)

def play_stop(timer1):
	global play_status
	timer1.stop()
	play_status = False
	print('playing status: %s' % play_status)

def play_start(timer1): 
	global play_status
	timer1.start(0.1)
	play_status = True
	print('playing status: %s' % play_status)

@canvas.connect
def on_key_press(event):
	global play_status, step, view_to_play, line1
	if event.text == ' ' and play_status == False:
		xlim = get_xlim(view_to_play)
		step  = (xlim[1]-xlim[0])/20.0
		play_start(timer1)

	elif event.text == ' ' and play_status == True:
		play_stop(timer1)

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
		# print get_xlim(view_to_play)
	elif event.text == 'y':
		ylim = get_ylim(view_to_play)
		print ylim
		# print get_ylim(view_to_play)
	elif event.text == 'n':
		N = 1000
		pos = np.zeros((N,2), dtype=np.float)
		pos[:,0] = np.linspace(0, 12, N)
		pos[:,1] = np.cos(pos[:,0]) + np.random.randn(len(pos[:,0]))
		line1.set_data(pos=pos)
		view_to_play.camera.set_range((0,pos[-1,0]))
	elif event.text == 'z':
		print view2.camera.azimuth
		print view2.camera.elevation
		print view2.camera.roll
		print view2.camera.distance

@canvas.connect
def on_mouse_double_click(event):
	global view_to_play
	print event.pos
	view2.camera.set_range()
	view3.camera.set_range()

def pos2d_to_pos3d(dist, cam):
    """Convert mouse x, y movement into x, y, z translations"""
    rae = np.array([cam.roll, cam.azimuth, cam.elevation]) * np.pi / 180
    sro, saz, sel = np.sin(rae)
    cro, caz, cel = np.cos(rae)
    print rae
    dx = (+ dist[0] * (cro * caz + sro * sel * saz)
          + dist[1] * (sro * caz - cro * sel * saz))
    dy = (+ dist[0] * (cro * saz - sro * sel * caz)
          + dist[1] * (sro * saz + cro * sel * caz))
    dz = (- dist[0] * sro * cel + dist[1] * cro * cel)
    return dx, dy, dz

@canvas.connect
def on_mouse_press(event):
	modifiers = event.modifiers
	button = event.button
	pos = event.pos
	if modifiers is not ():
		mod = [key.name for key in event.modifiers]
		if mod == ['Control']:
			print(mod,button,pos)
			cam2 = view2.camera
			# print cam2._viewbox.size
			# print cam2._viewbox.pos
			# print cam2._viewbox.margin
			center_x = cam2._viewbox.pos[0] + cam2._viewbox.size[0]/2
			center_y = cam2._viewbox.pos[1] + cam2._viewbox.size[1]/2
			print str(cam2.get_state())
			print (center_x,center_y)

			if cam2._event_value is None or len(cam2._event_value) == 2:
				cam2._event_value = cam2.center
			if is_in_view(pos, cam2):
				dist = pos - (center_x, center_y)
				dist[1] *= -1
				# Black magic part 1: turn 2D into 3D translations
				x,y,z = pos2d_to_pos3d(dist,cam2)
				# Black magic part 2: scale for mapping exact mouse event pos
				scale = 1.48 * cam2._scale_factor / init_scale_factor
				# Black magic part 3: take up-vector and flipping into account
				c = cam2.center
				ff = cam2._flip_factors
                up, forward, right = cam2._get_dim_vectors()
                x, y, z = right * x + forward * y + up * z
                x, y, z = ff[0] * x, ff[1] * y, z * ff[2]
                x,y,z = c[0] + scale*x, c[1] + scale*y, c[2] + scale*z
                iso.transform = scene.transforms.STTransform(translate=(x,y,z))

timer1 = app.Timer()
timer1.connect(play_trace)

###############################################################
if __name__ == '__main__':
	canvas.app.run()
