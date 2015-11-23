from pylab import *
from scipy.signal import filtfilt, butter, resample
import numpy as np

# Dynamically assign array
# Example:
# a = marray([0])
# a[0]=5.4
# a[1]=3.6
# a[2]=4.8
# a = np.array(a)
class marray(np.ndarray):

    def __setitem__(self, key, value):

        # Array properties
        nDim = np.ndim(self)
        dims = list(np.shape(self))

        # Requested Index
        if type(key)==int: key=key,
        nDim_rq = len(key)
        dims_rq = list(key)

        for i in range(nDim_rq): dims_rq[i]+=1        

        # Provided indices match current array number of dimensions
        if nDim_rq==nDim:

            # Define new dimensions
            newdims = []
            for iDim in range(nDim):
                v = max([dims[iDim],dims_rq[iDim]])
                newdims.append(v)

            # Resize if necessary
            if newdims != dims:
              self.resize(newdims,refcheck=False)

        return super(marray, self).__setitem__(key, value)

# Create an order 3 bandpass butterworth filter.
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def peakdet(v, delta, x = None, negthr = 0, posthr = 0):
    maxtab = []
    mintab = []
    if x is None:
        x = arange(len(v))
    v = asarray(v)
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta and this > posthr:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta and this < negthr:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)

# intracellular spike detection (in&ex extraction)
def get_spk(t, intra_trace, extra_trace):
    _t_pre = 0.0040
    _t_post = 0.0010
    spk_epoch = []
    spk_wave_intra = []
    spk_wave_extra = []
    # N = 2 # down sampling
    # intra_trace_d, t_d = resample(intra_trace, intra_trace.shape[0]/N, t)
    maxtab, _ = peakdet(intra_trace, delta=3, x=t, negthr=0, posthr=-20)
    for t0 in maxtab[:,0]:
        _t = logical_and(t>t0-_t_pre, t<t0+_t_post)
        spk_epoch.append(where(_t)[0])
        spk_wave_intra.append(intra_trace[_t])
        spk_wave_extra.append(extra_trace[_t])
    return array(spk_epoch), array(spk_wave_intra), array(spk_wave_extra)


# extracellular spike detection
def spkdet(trace, thr, t):
    t_post = 0.88e-3
    t_pre = 0.88e-3
    _, mintab = peakdet(trace, 1, t, -thr)
    spk = []
    spk_pt = []
    spk_pv = []
    spk_t = []
    for peak_t,peak_v in mintab:
        spk_duration = logical_and(t-peak_t<t_post,t-peak_t>-t_pre)
        spk.append(trace[spk_duration])     # spike traces
        spk_t.append(where(spk_duration))   # spike durations (exact time duration)
        spk_pt.append(peak_t)               # spike peak timings
        spk_pv.append(peak_v)               # spike peak amplitudes
    return spk,spk_t,spk_pt,spk_pv

####################################################################################
########################## Igor Class #####################
# example:
# ig = Igor()
# ig.load('***.pxp')
# ig.get_id()
# ig.sort()
# t, y1, y2 = ig[0]


import igor.igorpy as igor
import os
import re
import numpy as np
igor.ENCODING = 'UTF-8'

class Igor():
    """
    A interface to read igor traces"""
    def __init__(self, igorfile=None):
        self.im_path = None
        self._im_files = {}
        self.data = {}
        self._id_list = []
        if igorfile is not None:
            self.load(igorfile)

    # load igor file
    def load_file(self, igorfile):
        self.igorfile = igorfile
        self.igor_datum = igor.load(igorfile)
        print('load %s' % igorfile)
        self.igor_list = dir(self.igor_datum)
        print(self.igor_list)
        # self.get_id()
        # print('get_id() to get id list')

    def load_data(self):
        for _id in self._id_list:
            extra_id_0 = _id + '1_Ax1_Vm' # Axon
            extra_id_1 = _id + '1_Ch2_Vm' # Dendrite
            intra_id = _id + '1_Ch1_Vm'
            try:
                igor_trace = self.igor_datum[_id]
                t = igor_trace[intra_id].axis[0] * 1e6
                t = t[::-1]
                intra_trace = igor_trace[intra_id].data * 1e3 # unit(mV)
                extra_trace = np.zeros((len(t),2))
                extra_trace[:,0] = igor_trace[extra_id_0].data * 1e6 # unit(uV) Axon (left)
                extra_trace[:,1] = igor_trace[extra_id_1].data * 1e6 # unit(uV) Dendrite (right)
                self.data[_id] = (_id, t, intra_trace, extra_trace)
            except:
                pass

    # get id from intersction of igor file and imgpath
    def get_id_from_img(self, impath=None):
                  # './data/2015-10-16/invivo4.pxp'
        # im_path = "./data/2015-10-16/invivo4/"
        if impath is None:
            im_path = self.igorfile[:-4] + '/'
        else:
            im_path = impath
        # print('search in %s' % im_path)
        self.im_path = im_path

        # get _im_files in the folder as list
        for filename in os.listdir(im_path):
            if filename.endswith(".tiff"):
                self._im_files[filename.split('_')[0]] = [filename]
        # get id list
        id_list = [i for i in self._im_files.keys() if i in self.igor_list]
        for _id in id_list:
            self._id_list.append(_id)
        # self.id()

    # print current id list
    def id(self):
        print('id=%s' % self._id_list)

    # sort id list first with alphabit, then with number
    def sort(self):
        id_list = self._id_list
        n = []
        num = []
        for _d in id_list:
            n.append(re.search('\d',_d).start())
            num.append(int(filter(str.isdigit, _d)))
        _n = iter(n)
        _num = iter(num)
        self._id_list = sorted( id_list, key=lambda _id: (_id[:_n.next()], _num.next()) )

    
    def __getitem__(self, key):
        if isinstance(key, int):
            # print key
            _id = self._id_list[key]
            return self.data[_id]

        elif isinstance(key, str):
            # print 'str:' + key
            _id = key

            return _id, t, intra_trace, extra_trace

    def __iter__(self):
        for _id in self._id_list:
            extra_id_0 = _id + '1_Ax1_Vm' # Axon
            extra_id_1 = _id + '1_Ch2_Vm' # Dendrite
            intra_id = _id + '1_Ch1_Vm'
            igor_trace = self.igor_datum[_id]
            t = igor_trace[intra_id].axis[0] * 1e6
            t = t[::-1]
            intra_trace = igor_trace[intra_id].data * 1e3 # unit(mV)
            extra_trace = np.zeros((len(t),2))
            extra_trace[:,0] = igor_trace[extra_id_0].data * 1e6 # unit(uV) Axon (left)
            extra_trace[:,1] = igor_trace[extra_id_1].data * 1e6 # unit(uV) Dendrite (right)
            return _id, t, intra_trace, extra_trace

    def __len__(self):
        return len(self._id_list)

    def __repr__(self):
        if self.im_path is None:
            return str(self.igorfile) + '\n' + 'Warn: no id list, use get_id()'
        else:
            return str(self.im_path) + '\n' + str(self._id_list)


#########################################################################################################
# %load ipynb_snippets/open_file_ipython.py
import Tkinter,tkFileDialog
from IPython.html import widgets
from IPython.display import display

class open_file_btn(object):
    """IPython widgets for open_file_btn
       Gives you a row of buttons by:
       open_file_btn(['file','file','folder'])
       The chosen file is self.filename
       The chosen folder is self.foldername
    """

    def open_file(self, sender):
        root = Tkinter.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        filename = tkFileDialog.askopenfile(parent=root)
        self.filename.append(filename.name)
        print 'filename = %s' % filename.name
        root.destroy()

    def open_folder(self, sender):
        root = Tkinter.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        foldername = tkFileDialog.askdirectory(parent=root) + '/'
        self.foldername.append(foldername)
        print 'foldername = %s' % foldername
        root.destroy()

    def __init__(self, btn_structure=['file']):
        self.btns = []
        self.filename = []
        self.foldername = []

        for name in btn_structure:
            if name == 'file':
                btn = widgets.Button(description = 'Choose File')
                btn.button_style = 'Info'
                btn.width =120
                btn.height = 60
                btn.border_radius = 15
                btn.margin = 15
                btn.on_click(self.open_file)
                self.btns.append(btn)
            elif name == 'folder':
                btn = widgets.Button(description = 'Choose Folder')
                btn.button_style = 'Primary'
                btn.width = 120
                btn.height = 60
                btn.border_radius = 15
                btn.margin = 15
                btn.on_click(self.open_folder)
                self.btns.append(btn)
        display(widgets.HBox((self.btns)))

    def __repr__(self):
        return 'filename = '   + str(self.filename) + '\r\n' + \
               'foldername = ' + str(self.foldername)


####### decoration for jupyter button function ##########################################################
def btn_func(description='deco', color='lightlime'):
    def btn_deco(func):
        # description='deco'
        # color='#495ba3'
        btn = info_btn(description = description, color=color)
        btn.on_click(func)
        btn.show()
    return btn_deco   # this is necessary, so super-deco create deco

class info_btn(widgets.Button):
    """docstring for info_btn"""
    def __init__(self, description = '', color=''):
        super(info_btn, self).__init__()
        self.description = description
        self.button_style = 'Info'
        self.width = 200
        self.height = 60
        self.border_radius = 15
        self.font_size = 18
        self.margin = 15
        if color is not '':
            self.background_color=color

    def show(self):
        display(self)


#########################################################################################################

# btn.background_color = '#49cbd3'
# btn.border_color='#49cbd3'


if __name__ == '__main__':
    spk_epoch, spk_wave_intra, spk_wave_extra = get_spk(t, intra_trace, extra_trace)
    spk_wave_intra = spk_wave_intra.swapaxes(0,1);
    spk_wave_extra = spk_wave_extra.swapaxes(0,1);
    subplot(211); plot(spk_wave_intra,'b');
    subplot(212); plot(spk_wave_extra[:,:,0],'r'); plot(spk_wave_extra[:,:,1],'k')
