import numpy as np
import ruamel.yaml as yaml


class ShapeInfo:
	shape = ()
	
	def __init__(self, shape=None):
		self.shape = shape
	
	@property
	def n_bins(self):
		return self.shape[0]

	@property
	def n_rows(self):
		return self.shape[1]

	@property
	def n_cols(self):
		return self.shape[2]


def make_stim_info(cfg):
	dt = cfg['sim']['dt']
	span_info = cfg['stim']['span']
	size_info = cfg['stim']['size']
	nx = size_info['x']
	ny = size_info['y']
	try:
		nt = size_info['t']
	except KeyError:
		nt = int(span_info['t'] / dt)
	shp = ShapeInfo((nt, ny, nx))
	shp.axis_t = np.arange(0, span_info['t'], dt)
	shp.axis_x = ax_x = np.linspace(*span_info['x'], nx)
	shp.axis_y = ax_y = np.linspace(*span_info['y'], ny)
	gx, gy = np.meshgrid(ax_x, ax_y)
	shp.grid_x = gx
	shp.grid_y = gy
	return shp


def make_filt_info(cfg):
	dt = cfg['sim']['dt']
	stim_info = cfg['stim']
	span_info = cfg['filt']['span']
	size_info = cfg['filt']['span']
	nx = size_info.get('x', stim_info['size']['x'])
	ny = size_info.get('y', stim_info['size']['y'])
	try:
		nt = size_info['t']
	except KeyError:
		nt = int(span_info['t'] / dt)
	shp = ShapeInfo((nt, ny, nx))
	shp.axis_t = dt + np.arange(0, span_info['t'], dt)
	shp.axis_x = ax_x = np.linspace(
			*span_info.get('x', stim_info['span']['x']), nx)
	shp.axis_y = ax_y = np.linspace(
			*span_info.get('y', stim_info['span']['y']), ny)
	gx, gy = np.meshgrid(ax_x, ax_y)
	shp.grid_x = gx
	shp.grid_y = gy
	return shp


def load(filename):
	with open(filename, 'r') as cfg_file:
		cfg = yaml.load(cfg_file, Loader=yaml.Loader)
		stim = make_stim_info(cfg)
		filt = make_filt_info(cfg)
		cfg['stim'] = stim
		cfg['filt'] = filt
	return cfg
