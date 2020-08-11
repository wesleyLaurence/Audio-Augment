# -*- coding: utf-8 -*-
"""
Created on Tues Aug 10 12:42:34 2020

@author: Wesley Laurence
"""

# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import soundfile as sf
import librosa
from scipy import signal
from scipy.signal import butter, lfilter, freqz

import IPython.display as ipd
from pathlib import Path
import os
import sys
import warnings

# default plotting style
plt.style.use('seaborn-dark-palette')

class AudioAugment:   
    """ AudioAugment is a data augmentation tool designed to expand audio datasets for machine learning projects. """

    def __init__(self):
        """ initialize """           
    
    
    def sample(self, file_name):
        """ Load WAV file from unprocssed directory.
        
        Input
            file_name: file name of WAV file you would like to load (ie `my_audio.wav`)
        
        Output
            Wave object, which is the audio data stored in an array
        """
        
        # WAV file path
        file_path = "WAV/unprocessed/"+file_name
        
        # load WAV file using soundfile
        data, sr = sf.read(file_path)    
        
        # convert to Wave object
        self.waveform = Wave(data)
  
        return self.waveform


    def augment_dataset(
        self, 
        HPF_cutoff=700, 
        LPF_cutoff=1300, 
        noise_level=0.02,
        low_factor = 1.3,
        high_factor = 0.7,
        slow_factor = 1.3,
        fast_factor = 0.7
    ):
        """ This algorithm augments audio training datasets by transforming the provided WAV files in 10 different ways:
        
            1. Waveform Inversion
            2. Highpass Filter
            3. Lowpass Filter
            4. Bandpass Filter
            5. Add noise (normal, uniform)
            6. Pitch shift (low, high)
            7. Time shift (slow, fast)
        
        
        Directory Setup:
            Create a folder in your working directory called `WAV`. Then create two subfolders inside WAV, 
            one called `unprocessed` and another called `processed`.
            Place all audio samples that you want to transform in the unprocessed folder (must be WAV format). 
            Run this method and it will transform all audio files in your unprocessed folder, 
            and save the new set of augmented samples in the processed folder.
        
        
        Optional Input Parameters:
            - HPF_cutoff: desired cutoff frequency for the high pass filter transformation (also affects bandpass filter)
            
            - LPF_cutoff: desired cutoff frequency for the low pass filter transformation (also affects bandpass filter)
            
            - noise_level: controls the volume of the normal & uniform noise added to the signal
            
            - low_factor: determines how low the pitch is for the down-shift. Increase this value to make the pitch shift lower.
            
            - high_factor: determines how high the pitch is for the up-shift. Decrease this value to make the pitch shift higher.
            
            - slow_factor: determines how slow the time-shift is. Increase this value to make the time shift slower.
            
            - fast_factor: determines how fast the time-shift is. Decrease this value to make the time shift faster.

        """
        # get list of all wav files in unprocessed directory
        wav_files = os.listdir("WAV/unprocessed")
        
        # iterate through all files in unprocessed directory
        for wav_file in wav_files:
            
            # only process WAV files
            if wav_file[-4:] == ".wav":
         
                # load wav file, create wave object
                waveform = self.sample(wav_file)

                # if file is stereo, convert to mono
                if waveform.ndim == 2:
                    waveform = Wave(waveform.T[0] + waveform.T[1])

                    # Prevent clipping after sum to mono
                    while abs(waveform).max() < .5:
                        waveform = Wave(waveform*1.25)

                    while abs(waveform).max() > 1:
                        waveform = Wave(waveform*.75)

                # original 
                original_waveform = Wave(waveform)
                original_filename = wav_file[:-4]+"_original.wav"
                original_waveform.bounce(original_filename, show_visual=False)

                # inversion 
                inverted_waveform = Wave(waveform * -1)
                inverted_filename = wav_file[:-4]+"_inverted.wav"
                inverted_waveform.bounce(inverted_filename, show_visual=False)

                # LPF 
                LPF_waveform = Wave(waveform.LPF(LPF_cutoff))
                LPF_filename = wav_file[:-4]+"_LPF.wav"
                LPF_waveform.bounce(LPF_filename, show_visual=False)

                # HPF 
                HPF_waveform = Wave(waveform.HPF(HPF_cutoff))
                HPF_filename = wav_file[:-4]+"_HPF.wav"
                HPF_waveform.bounce(HPF_filename, show_visual=False)

                # Bandpass 
                bandpass_waveform = Wave(waveform.HPF(HPF_cutoff).LPF(LPF_cutoff))
                bandpass_filename = wav_file[:-4]+"_bandpass.wav"
                bandpass_waveform.bounce(bandpass_filename, show_visual=False)

                # add normal distribution noise
                sample_len = len(waveform)
                normal_noise = np.random.normal(0, noise_level, sample_len)
                normal_noise_waveform = Wave(waveform + normal_noise)
                normal_noise_filename = wav_file[:-4]+"_normalNoise.wav"
                normal_noise_waveform.bounce(normal_noise_filename, show_visual=False)

                # add uniform distribution noise
                uniform_noise = np.random.uniform(0, noise_level, sample_len)
                uniform_noise_waveform = Wave(waveform + uniform_noise)
                uniform_noise_filename = wav_file[:-4]+"_uniformNoise.wav"
                uniform_noise_waveform.bounce(uniform_noise_filename, show_visual=False)

                # low pitch 
                low_waveform = librosa.core.resample(waveform, sample_len, round(sample_len * low_factor))
                low_waveform = Wave(low_waveform)
                low_filename = wav_file[:-4]+"_low.wav"
                low_waveform.bounce(low_filename, show_visual=False)

                # high pitch
                high_waveform = librosa.core.resample(waveform, sample_len, round(sample_len * high_factor))
                high_waveform = Wave(high_waveform)
                high_filename = wav_file[:-4]+"_high.wav"
                high_waveform.bounce(high_filename, show_visual=False)

                # slow timestretch
                slow_waveform = librosa.effects.time_stretch(waveform, rate=sample_len/round(sample_len * slow_factor))
                slow_waveform = Wave(slow_waveform)
                slow_filename = wav_file[:-4]+"_slow.wav"
                slow_waveform.bounce(slow_filename, show_visual=False)

                # fast timestretch
                fast_waveform = librosa.effects.time_stretch(waveform, rate=sample_len/round(sample_len * fast_factor))
                fast_waveform = Wave(fast_waveform)
                fast_filename = wav_file[:-4]+"_fast.wav"
                fast_waveform.bounce(fast_filename, show_visual=False)

        return print("Dataset successfully augmented!")
            
            
class Wave(np.ndarray):
    """
    The wave object is the primary data structure for audio waveforms in AudioAugment. Wave is a sub class of a numpy array. The wave object has the & efficiency of a numpy array and is extended with custom methods for audio functionality.
    """
    
    # initialize sub class of np.array
    def __new__(cls, waveform):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(waveform).view(cls)
        # add the new attribute to the created instance
        obj.waveform = obj
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.waveform = getattr(obj, 'waveform', None)
       
    # Initialize all variables and import data
    def __init__(self, waveform):
        self.waveform = waveform
        
        # default sample rate
        self.Fs = 44100 
     
    # export WAV to directory
    def bounce(self, file_name='user_playback.wav', sample_rate=44100, show_visual=True):
        """ This method exports a bounced WAV file to the unprocessed directory """
        
        data = self.waveform
        
        wav_files = os.listdir("WAV/unprocessed")

        if file_name in wav_files:
            file_name = file_name[:-4] + '_'+str(1)+'.wav'
        else:
            pass

        i=1
        while file_name in wav_files:
            i+=1
            file_name_items = file_name[:-4].split('_')
            iteration_num =int(file_name[:-4].split('_')[-1])
            iteration_num+=1
            file_name_items[-1] = str(iteration_num)
            file_name = '_'.join(file_name_items)+'.wav'
 
        length_seconds = data.shape[0]/sample_rate
        file_path = 'WAV/processed/'+file_name
        
        sf.write(file_path, data, sample_rate)        
        
        # show waveform 
        if show_visual == True: 
            # if waveform is stereo...
            if self.waveform.T.shape[0] == 2:
                wave_l = self.waveform.T[0]
                wave_r = self.waveform.T[1]
                fig, axs = plt.subplots(2, sharex=True, sharey=True,figsize=(700/96, 200/96), dpi=96)
           
                if 'user_playback' in file_name:
                    pass
                else:
                    fig.suptitle(file_name)
                    
                axs[0].plot(wave_l, color='mediumblue')
                axs[1].plot(wave_r, color='mediumblue')
                plt.show()

            else:
                # if waveform is mono...
                wave_mono = self.waveform[0]

                wave_mono = self.waveform
                fig, axs = plt.subplots(1,figsize=(700/96, 200/96), dpi=96)
                
                if 'user_playback' in file_name:
                    pass
                else:
                    fig.suptitle(file_name)

                axs.plot(wave_mono,color='mediumblue')
                plt.show()     
        else:
            pass
    
        return ipd.Audio(file_path)
    
    
    def view(self, plot_type='all' ,title=None, freq_range=(20,20000)):
        """ add method description here """
        
        # WAVE VISUAL functions

        def wave_view(waveform):
            fig,ax = plt.subplots(figsize=(12, 4),dpi=96)
            plt.subplot(212)
            plt.plot(waveform,color='mediumblue')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.title('Waveform')
            plt.show()

        def spectrogram(waveform):
            """ add function description here """
            signalData = waveform 
            samplingFrequency = 44100

            fig,ax = plt.subplots(figsize=(12, 4),dpi=96)
            plt.subplot(212)
            plt.specgram(signalData,Fs=samplingFrequency)
            plt.ylim(freq_range[0],freq_range[1])
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title('Spectrogram')
            plt.show()

        def spectrum(waveform):
            """ add function description here 
            
            SOURCE: https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
            """
            
            # sampling information
            Fs =self.Fs # sample rate
            N = len(waveform)
            t_vec = np.arange(N) # time vector for plotting


            # fourier transform and frequency domain
            #
            Y_k = np.fft.fft(waveform)[0:int(N/2)]/N # FFT function from numpy
            Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
            Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

            f = Fs*np.arange((N/2))/N; # frequency vector

            # plotting
            fig,ax = plt.subplots(figsize=(12, 4),dpi=96)
            plt.plot(f,Pxx,linewidth=1,color='mediumblue')
            ax.grid(color='w', linestyle='-', linewidth=1, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.title('Spectrum')
            plt.ylabel('Amplitude')
            plt.ylim(0.01,1)
            plt.xlabel('Frequency [Hz]')
            plt.xlim(freq_range[0],freq_range[1])
            plt.show()
        
        
        # if waveform is stereo
        if self.waveform.ndim == 2:
            wave_l = self.waveform.T[0]
            wave_r = self.waveform.T[1]
            summed_waveform = wave_l + wave_r

            if 'all' in plot_type:
                wave_view(summed_waveform)
                spectrum(summed_waveform)
                spectrogram(summed_waveform)
                
            elif 'wave' in plot_type:
                wave_view(summed_waveform)
            elif 'spectro' in plot_type:
                spectrogram(summed_waveform)
            elif 'spectrum' in plot_type:
                spectrum(summed_waveform)
         
        # if waveform is mono
        else:
            
            if 'all' in plot_type:
                wave_view(self.waveform)
                spectrum(self.waveform)
                spectrogram(self.waveform)
            elif 'wave' in plot_type:
                wave_view(self.waveform)
            elif 'spectro' in plot_type:
                spectrogram(self.waveform)
            elif 'spectrum' in plot_type:
                spectrum(self.waveform)
                       
        duration = round(self.waveform.shape[0]/self.Fs, 3)    
 
    # LOW PASS FILTER
    def LPF(self, cutoff, order=5):
        """ add method description here """
        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, self.waveform)
            return y

        if self.waveform.ndim == 1: 
            data = self.waveform
            # Filter the data, and plot both the original and filtered signals.
            filtered_wave = butter_lowpass_filter(data, cutoff, self.Fs, order)
            waveform = filtered_wave
            
        else: 
            unfiltered_l = self.waveform.T[0]
            unfiltered_r = self.waveform.T[1]
            
            filtered_l = butter_lowpass_filter(unfiltered_l,cutoff,self.Fs,order)
            filtered_r = butter_lowpass_filter(unfiltered_r,cutoff,self.Fs,order)
            
            waveform = np.array([filtered_l,filtered_r]).T
            
        # while volume is quiet, gradually make louder until threshold is reached
        while abs(waveform).max() < .5:
            waveform = waveform*1.25

        # while volume is clipping, make quiter
        while abs(waveform).max() > 1:
            waveform = waveform*.75
       
        self.waveform = Wave(waveform)
       
        return self.waveform
    
    
    # HIGH PASS FILTER
    def HPF(self, cutoff, order=5):
        """ add method description here """
        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def butter_highpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = signal.filtfilt(b, a, self.waveform)
            return y

        if self.waveform.ndim == 1: 
            data = self.waveform
            # Filter the data, and plot both the original and filtered signals.
            filtered_wave = butter_highpass_filter(data, cutoff, self.Fs, order)
            
        elif self.waveform.ndim == 2:
            unfiltered_l = self.waveform.T[0]
            unfiltered_r = self.waveform.T[1]
            
            filtered_l = butter_highpass_filter(unfiltered_l,cutoff,self.Fs,order)
            filtered_r = butter_highpass_filter(unfiltered_r,cutoff,self.Fs,order)
            
            filtered_wave = np.array([filtered_l,filtered_r]).T
            
        # hide weird warning about multi dimensional tuple indexing cause... IDK that is    
        def fxn():
            warnings.warn("deprecated", DeprecationWarning)
            
        # while volume is quiet, gradually make louder until threshold is reached
        while abs(filtered_wave).max() < .5:
            filtered_wave = filtered_wave*1.25

        # while volume is clipping, make quiter
        while abs(filtered_wave).max() > 1:
            filtered_wave = filtered_wave*.75
        
        self.waveform = Wave(filtered_wave)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
        
        return self.waveform