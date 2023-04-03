#Programmed by: Rey Benjamin M. Baquirin
#This program goes through the steps of extrating MFCCs and provides visualization per vital step
#It takes in an input wav file for demonstration

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import sigproc,base
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import magnitude_spectrum
import pandas as pd

if __name__=='__main__':

	input_file = "./words_visualize/computer_1.wav"

	# Read the input file
	sampling_freq, audio = wavfile.read(input_file)

	#Plot Raw Audio
	plt.title('Raw Input Signal, computer_1.wav')
	plt.plot(audio, linewidth=0.5)
	plt.ylabel('Amplitude (dB)')
	plt.xlabel('Time (seconds)')
	plt.show()

	frames = sigproc.framesig(audio, 0.025*sampling_freq, 0.01*sampling_freq, winfunc=np.hamming)
	plt.title('Hamming Window, 25ms, computer_1.wav')
	plt.plot(np.arange(0, 25, 25/200),frames[0], linewidth=1)
	plt.ylabel('Amplitude (dB)')
	plt.xlabel('Time (milliseconds)')
	plt.show()

	signal = sigproc.preemphasis(frames[0], 0.95)
	plt.title('The Spectrum')
	spec = magnitude_spectrum(signal, sampling_freq*0.05, linewidth=1)
	plt.xlabel('Frequency (Hz)')
	plt.show()

	filters = base.get_filterbanks(nfilt=26, nfft=402, samplerate=sampling_freq).T
	plt.title('Mel Filterbank with 26 filters')
	plt.plot(filters,linewidth=0.5)
	plt.xlabel('Frequency (Hz)')
	plt.show()

	mel_spectrum = base.fbank(spec[0], sampling_freq, winlen=0.00025, winstep=0.0001, nfilt=26, nfft=402, preemph=0.97, winfunc=np.hamming)
	plt.title('Mel Scale Filtering of the Spectrum')
	mel_spec_feats = np.array([])
	mel_spec_feats = mel_spectrum[0]
	mel_spec_feats= np.insert(mel_spec_feats, 0, 0, axis=0)
	plt.plot(spec[1],mel_spec_feats,linewidth=0.3)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude (energy) ')
	plt.show()

	plt.title('Mel Frequency Spectrum')
	mel_spec_feats2 = np.array([])
	mel_spec_feats2 = mel_spectrum[1]
	mel_spec_feats2 = np.insert(mel_spec_feats2, 0, mel_spec_feats2[0], axis=0)
	plt.plot(spec[1],mel_spec_feats2,linewidth=1)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude (energy) ')
	plt.show()

	log_mel = np.log(mel_spectrum[1])
	plt.title('Log of the Mel Frequency Spectrum')
	log_spec_feats = np.array([])
	log_spec_feats = log_mel
	log_spec_feats= np.insert(log_spec_feats, 0, log_spec_feats[0], axis=0)
	plt.plot(spec[1],log_spec_feats, linewidth=1)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Magnitude (energy) ')
	plt.show()

	cepstrum = fft.dct(log_mel)
	cepfeats = np.array([])
	cepfeats = cepstrum
	for i in range(100):
		cepfeats= np.append(cepfeats, 0)
	plt.plot(np.arange(0, 25, 25/200),cepfeats.T, linewidth=1)
	plt.title('The Cepstrum')
	plt.ylabel('Amplitude (dB)')
	plt.xlabel('Time (milliseconds)')
	plt.show()

	plt.show()


	final_mfcc = base.mfcc(audio, sampling_freq, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=402, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
	featheaders = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']

	df = pd.DataFrame(final_mfcc, columns=featheaders)

	firstframe = df.iloc[[0]]
	firstframe = firstframe.values.flatten()

	plt.title('The 13 MFCCs of the Cepstrum')
	plt.scatter(np.arange(13),firstframe, color=['aqua', 'black', 'azure', 'brown', 'chartreuse', 'crimson', 'gold', 'green', 'indigo', 'khaki', 'orange', 'red', 'blue'])
	plt.ylabel('MFCC Value')
	plt.xlabel('Coefficient Number')

	counts = np.arange(13)
	
	for i, txt in enumerate(featheaders):
    		plt.annotate(txt, (counts[i],firstframe[i]))

	plt.plot(np.arange(13), firstframe, linewidth=1)
	plt.show()

	counter = 0
	featarr = np.array([])

	for frames in df:
		frame = df.iloc[[counter]]
		featarr = np.append(featarr,frame.values)
		counter = counter + 1

	featarr =  featarr.reshape(counter, 13)
	for feats in featarr:
		plt.plot(counts, feats)

	plt.title('The first 13 MFCCs of the whole .wav file across all frames')
	plt.ylabel('MFCC Value')
	plt.xlabel('Coefficient Number')
	plt.show()