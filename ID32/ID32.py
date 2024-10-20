'''
TedMercer
This is a data loader for the beamline ID32 at the ESRF synchrotron
This was made in experiment and may have bugs 
This is object oriented and is meant to be used in a jupyter notebook / google colab notebook
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import interpolate
import json
import csv
import itertools
import h5py
import copy
from pathlib import Path
import sys
from PIL import Image
import re
import time
import os
import h5py
from scipy.interpolate import interp1d
import copy
from numpy.fft import fft, ifft, fftshift, ifftshift

# Must install silx to use this
from silx.io.specfile import SpecFile, Scan

# Utility functions
def shift_fourier( signals, dxs ):
    # Shift signal in Fourier space to (not) deal with interpolation
    # Signal is assume to be (ns,nx) or (ns,) array
    if signals.ndim == 1:
        signals = np.array( [signals] )

    dxs=np.asarray(dxs)
    if dxs.ndim == 0:
        dxs = [dxs]
    dxs = np.round( dxs, 2)
    print(dxs)

    ns, nx = signals.shape
    center = np.floor(nx/2)
    x = np.arange(0, nx)

    # Construcut K space grid: Very important to get this correct
    kx = (x-center)*(2*np.pi/(nx-1))
    signals_fft = fftshift( fft( signals), axes=-1 )
    signals_shifted = np.zeros( (ns,nx) )
    for i in range(0,ns):
        # Apply phase and bring the signal back to real space
        signals_fft_shifted = signals_fft[i,:]* np.exp( -1j*kx*dxs[i] )
        signals_shifted[i,:] = np.real( ifft( ifftshift(signals_fft_shifted, axes=-1 ) ))

    return signals_shifted

def find_shift_CC( signals, bool_plot=False, sx0=10, wd=10):
    # Cross correlate and find relative shift
    # Cross correlation is performed in Fourier Space
    # Shift is determined by fitting a Gaussian to Cross Correlation

    if signals.ndim == 1:
        signals = np.asarray( [signals] )

    ns, nx = signals.shape

    if bool_plot:
        fig, ax = plt.subplots(ns,1,figsize=(3,15))



    dx_fits = np.zeros( (ns) )
    # Pre-compute 1D FFTs
    signals_fft = fft( signals )
    # Compute Cross Correlations

    model = GaussianModel() + ConstantModel()

    for i in range(0,ns):
        cc = np.real( fftshift( ifft( np.conj( signals_fft[0,:]) * signals_fft[i,:] ) )) # fftshift rolls by nE[0]/2; easier to fit
        # print(cc)
        # Initial guess by locating cc max
        # This fails if elastic line is not max
        x0 = np.argmax(cc)


        params = Parameters()
        params.add('center', value=x0 )
        params.add('sigma' , value=sx0)
        params.add('amplitude', value=(np.max(cc)-np.min(cc)))
        params.add('c', value = np.min(cc))

        # Fit over a small window for better accuracy
        xs = np.arange(0, nx)
        xs_sub = xs[ (x0-wd):(x0+wd) ]
        cc_sub = cc[ (x0-wd):(x0+wd) ]

        # Fit a Gaussian to find better maximum
        result = model.fit(cc_sub,params,x=xs_sub)
        dx_fits[i] = ( result.params['center'].value -np.floor(nx/2) )


        if bool_plot:

            ax[i].plot( xs, cc, label='Cross-Correlation' )
            ax[i].plot( xs_sub,result.eval(x=xs_sub), label = 'Gaussian fit')
            ax[i].legend()
            ax[i].set_xlim( x0-2*wd, x0+2*wd )

    dx_fits = np.round( dx_fits, 2)
    return dx_fits

  class Spectra:
   def __init__(self,filename = '', num = 0, samplename = '', path='',metah5 = ''):
       self.filename = filename
       self.metah5 = path + metah5
       self.num = num
       self.path = path
       self.data = []
       self.samplename = samplename
       self.Emap = []
       self.Ei = []
       self.El = []
       self.SPC = []
       self.Mon = []
   def read_file(self, norm = 'acq', load_in = 'Energy', meta = False):
       if meta == True:
        key = str(self.num)+'.1'
        with h5py.File(self.metah5, 'r') as f:
          normfac = f[key]['measurement']['mir_rixs'][:]
        self.normfac = normfac

       sf = SpecFile(self.path + self.filename)
       sample = self.samplename
       maptype = 'energymap'
       offset = 0.05
       if self.num == 1:
           scno = str(self.num) + '.2'
       else:
           scno = str(self.num) + '.1'
       print(scno)
       try:
           sc = sf[scno]
           self.SPC = sc.data_column_by_name('SPC')
           if load_in == 'Energy':
            Eloss = -sc.data_column_by_name('Energy (auto)') - .0025
           if load_in == 'Pixel':
            Eloss = sc.data_column_by_name('Pixel')
           if norm == 'acq':
            self.Mon = sc.data_column_by_name('Acquisition time')
           if norm == 'else':
            self.Mon = sc.data_column_by_name('Acquisition time')
           if norm == 'I0':
            self.Mon = self.normfac
           En = sc.motor_position_by_name(name = 'energy')
           Counts_norm = self.SPC/self.Mon

           self.Emap.append(Counts_norm)
           self.Ei.append(En)
           self.El.append(Eloss)
           self.int = self.SPC/self.Mon
       except KeyError:
           print(f'No data for scan {self.num}...only completed {self.num-1} scans')
           pass
   def plot(self, offset = 0, rikkie = False, _label = None,_lw = 1.5, factor = 1,
            shift = 0,errorbars = False, yshift = 0, **kwargs ):
    if errorbars == False:
      if rikkie == False:
            plt.plot(self.El[0]+shift,self.int*factor + offset,**kwargs)
      else:
          plt.axhline(offset*(i-1), color='k', lw=0.75, zorder=0)
          plt.fill_between(self.El[0], self.El[1]*0 + offset+0.001, self.int + offset*(i-1), color='w')
          if _label == None:
            plt.plot(self.El[0]+shift,self.int*factor + offset,label = f'{np.round(self.Ei,2)}eV__{i}', **kwargs)
          else:
            plt.plot(self.El[0]+shift,self.int*factor + offset,label = _label,**kwargs)
    if errorbars == True:
      try:

        plt.errorbar(self.El[0]+shift, self.int*factor - yshift,yerr= self.error, **kwargs)
      except AttributeError:
        print("attribute does not exists")
        self.error = np.sqrt(self.SPC)/self.Mon
        plt.errorbar(self.El[0]+shift, self.int*factor - yshift,yerr= self.error, **kwargs)
    plt.legend()

   def shift_spectra(self, xshift = 0):
    C = copy.deepcopy(self)
    C.El[0] = C.El[0] + xshift
    return C

   def bin_data(self, bin_size):
    if len(self.El[0]) ==0 or len(self.int)==0:
      raise ValueError
      print('No data to BIN. PLEASE READ DATA FIRST! :o')
    num_bins = len(self.El[0]) // bin_size
    binned_El = np.mean(self.El[0][:num_bins*bin_size].reshape(-1, bin_size), axis = 1)
    binned_int = np.mean(self.int[:num_bins*bin_size].reshape(-1, bin_size), axis = 1)
    c = copy.deepcopy(self)
    c.El[0] = binned_El
    c.int = binned_int
    print(f'Data binned by {bin_size} -- obj copy is returned')
    return c

   def Pixels_to_Energy(self, res = 'hr'):
    if res == 'hr':
      slope = 7.222
      holder = 'HIGH RES'
    if res == 'lr':
      slope = 10.39
      holder = 'LOW RES'
    print(f'We are in {holder} mode with a slope of {slope} meV/Pixels')
    c = copy.deepcopy(self)
    c.El[0] = c.El[0]*slope / 1000
    print('now in energy (eV)....')
    maxidx = np.argmax(c.int)
    maxEl = c.El[0][maxidx]
    c.El[0] = -1*(c.El[0] - maxEl)
    print('now in energy (eV)....and centered')
    return c


   @staticmethod
   def avg_spectra(spectra_list):
       if not spectra_list:
        raise ValueError("The input list of Spectra objects is empty.")
       El_sum = np.zeros_like(spectra_list[0].El[0])
       int_sum = np.zeros_like(spectra_list[0].int)
       Ei_sum = np.zeros_like(spectra_list[0].Ei)
       Mon_sum = np.zeros_like(spectra_list[0].Mon)
       SPC_sum = np.zeros_like(spectra_list[0].SPC)
       error_sum = np.zeros_like(spectra_list[0].SPC)
       for spec in spectra_list:
         El_sum += spec.El[0]
         int_sum += spec.int
         Ei_sum += spec.Ei
         Mon_sum += spec.Mon
         SPC_sum += spec.SPC

       num_spectra = len(spectra_list)

       avg_El = El_sum / num_spectra
       avg_int = int_sum / num_spectra
       avg_Ei = Ei_sum / num_spectra
       avg_mon = Mon_sum / num_spectra
       avg_spc = SPC_sum / num_spectra
       error_avg = np.sqrt(avg_spc) / avg_mon

       avg_spectra = Spectra()
       avg_spectra.El = [avg_El]
       avg_spectra.int = avg_int
       avg_spectra.Ei = avg_Ei.tolist()
       avg_spectra.Mon = Mon_sum / num_spectra
       avg_spectra.SPC = SPC_sum / num_spectra
       avg_spectra.error = error_avg/np.sqrt(num_spectra)
       return avg_spectra

   @staticmethod
   def generate_dicroism(cpSpectra, cmSpectra, order = 'cp'):
    if not cpSpectra or not cmSpectra:
      raise ValueError("The input lists of Spectra objects are empty.")
    if order =='cp':
      int_dic = cpSpectra.int - cmSpectra.int
    if order =='cm':
      int_dic = cmSpectra.int - cpSpectra.int
    dic_spectra = Spectra()
    dic_spectra.El = cpSpectra.El
    dic_spectra.int = int_dic
    return dic_spectra

   @staticmethod
   def avg_fft(spectra_list):
    El_sum = np.zeros_like(spectra_list[0].El[0])
    Ei_sum = np.zeros_like(spectra_list[0].Ei)
    Mon_sum = np.zeros_like(spectra_list[0].Mon)
    SPC_sum = np.zeros_like(spectra_list[0].SPC)
    error_sum = np.zeros_like(spectra_list[0].SPC)
    for spec in spectra_list:
      El_sum += spec.El[0]
      Ei_sum += spec.Ei
      Mon_sum += spec.Mon
      SPC_sum += spec.SPC
    num_spectra = len(spectra_list)
    avg_El = El_sum / num_spectra
    avg_Ei = Ei_sum / num_spectra
    avg_mon = Mon_sum / num_spectra
    avg_spc = SPC_sum / num_spectra
    error_avg = np.sqrt(avg_spc) / avg_mon
    dx = np.zeros(num_spectra)
    data = np.asarray([dataset.int for dataset in spectra_list])
    fft_data = np.zeros_like(spectra_list[0].int)
    dx = find_shift_CC(data,bool_plot=False, sx0=10, wd=15)
    fft_data= shift_fourier(data,-dx)
    for data in fft_data:
      plt.plot(data)
    plt.xlim(1700,1900)
    avg_fft_data = np.mean(fft_data, axis=0)




    #fft_data = [fftshift(fft(dataset.int), axes = -1) for dataset in spectra_list]

    #avg_spectra = np.abs(ifft(ifftshift(avg_fft_data, axes = -1)))
    energyaxis = spectra_list[0].El[0]
    avg_fft_tot = Spectra()
    avg_fft_tot.Ei = avg_Ei[0]
    avg_fft_tot.Mon = Mon_sum / num_spectra
    avg_fft_tot.SPC = SPC_sum / num_spectra
    avg_fft_tot.error = error_avg/np.sqrt(num_spectra)
    avg_fft_tot.El = [energyaxis]
    avg_fft_tot.int = avg_fft_data
    print('Spectra has been created', dir(avg_fft_tot))
    return avg_fft_tot


class Spectrum:
   def __init__(self, spectra):
       self.spectra = spectra
       self.Emap = []
       self.Ei = []
       self.El = []
   def make_spectrum(self):
       for i,spectra in enumerate(self.spectra):
           self.Ei.append(spectra.Ei[0])
           self.El.append(spectra.El)
           self.Emap.append(spectra.Emap)
       print(f'Spectrum made with {len(self.spectra)} spectra')

   def make_color_map(self):
       step = -np.mean(np.diff(self.El[0]))
       self.x = np.arange(-6,16,step)
       self.y_interp = np.zeros((len(self.Ei),len(self.x)))
       for i,n in enumerate(self.Ei):
           self.y_interp[i,:] = np.interp(self.x, np.flip(self.El[i][0]), np.flip(self.Emap[i][0]))


   def plot_color_map(self, vmin = 0, vmax = 1, cmap = 'viridis'):
       plt.pcolormesh(self.Ei,self.x, self.y_interp.T, cmap = cmap, vmin = vmin, vmax = vmax)

   def compute_PFY(self):
       l_lim = np.where(self.x<=0.5)[0][-1]
       h_lim = np.where(self.x>=.25)[0][0]
       self.PFY_eline = np.sum(self.y_interp[:,l_lim:h_lim],axis = 1)
       l_lim = np.where(self.x<=1)[0][-1]
       h_lim = np.where(self.x>=5.5)[0][0]
       self.PFY_dd = np.sum(self.y_interp[:,l_lim:h_lim],axis = 1)


class XAS_spec:
    def __init__(self, filen, path):
        self.file = os.path.join(path, filen)

    def generate_spectra(self):
        """Generates spectra attributes from the HDF5 file."""
        with h5py.File(self.file, 'r') as f:
            self.energy = f['1.1']['measurement']['energy_enc'][:]
            self.TFY = f['1.1']['measurement']['dbig_rixs'][:]
            self.TFY_norm = f['1.1']['measurement']['dbig_n'][:]
            self.TEY = f['1.1']['measurement']['sam_rixs'][:]
            self.TEY_norm = f['1.1']['measurement']['sam_n'][:]
            self.meta = f['1.1']
        print('Spectra generated -- attributes: energy, TFY, TFY_norm, TEY, TEY_norm, meta')


    def display_hdf5_tree(self):
        """Displays the HDF5 structure in a tree format."""
        with h5py.File(self.file, 'r') as f:
            print("HDF5 File Structure:")
            self._print_hdf5_keys(f)


    def _print_hdf5_keys(self, group, indent=0):
        """Recursively prints HDF5 groups and datasets."""
        for key in group:
            item = group[key]
            print("    " * indent + f"├── {key}")
            if isinstance(item, h5py.Group):
                self._print_hdf5_keys(item, indent + 1)

    def plot_XAS(self, norm=True):
        """Plots the XAS spectra."""
        plt.figure()
        if norm:
            plt.plot(self.energy, self.TFY_norm, label='TFY (Normalized)')
            plt.plot(self.energy, self.TEY_norm, label='TEY (Normalized)')
        else:
            plt.plot(self.energy, self.TFY, label='TFY')
            plt.plot(self.energy, self.TEY, label='TEY')
        plt.legend()
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (arb. units)')
        plt.title('XAS Spectra')
        plt.show()


    def bin_data(self, bin_size):
        """Bins the energy and corresponding data arrays with a specified bin size."""
        if bin_size <= 0:
            raise ValueError("Bin size must be greater than 0")

        energy_binned = np.mean(self.energy[:len(self.energy)//bin_size * bin_size].reshape(-1, bin_size), axis=1)

        TFY_binned = np.mean(self.TFY[:len(self.TFY)//bin_size * bin_size].reshape(-1, bin_size), axis=1)
        TFY_norm_binned = np.mean(self.TFY_norm[:len(self.TFY_norm)//bin_size * bin_size].reshape(-1, bin_size), axis=1)
        TEY_binned = np.mean(self.TEY[:len(self.TEY)//bin_size * bin_size].reshape(-1, bin_size), axis=1)
        TEY_norm_binned = np.mean(self.TEY_norm[:len(self.TEY_norm)//bin_size * bin_size].reshape(-1, bin_size), axis=1)

        self.energy = energy_binned
        self.TFY = TFY_binned
        self.TFY_norm = TFY_norm_binned
        self.TEY = TEY_binned
        self.TEY_norm = TEY_norm_binned

        print(f"Data binned with bin size {bin_size}")


    def display_meta(self):
        """Prints metadata information from the HDF5 file."""
        print("Metadata information:")
        for key in self.meta.attrs:
            print(f"{key}: {self.meta.attrs[key]}")
