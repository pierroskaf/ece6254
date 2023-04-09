# Features extracted from the Librosa library
#  https://librosa.org/doc/main/feature.html

import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
from sklearn import *
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

from glob import glob
import os
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool

path = 'Data/genres_original/'
features = []

def load_song(song_path):
    x, sr = librosa.load(song_path)
    return x, sr

def calculate_mean(arr):
    return np.mean(arr)
    
def calculate_var(arr):
    return np.var(arr)

def get_tempo(x):
    return librosa.feature.tempo(y=x)[0]

def get_harmony(x):
    return librosa.effects.harmonic(y=x)

def get_perc_weight(x,sr):
    C = np.abs(librosa.cqt(y=x, sr=sr, fmin=librosa.note_to_hz('A1')))
    freqs = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz('A1'))
    return librosa.perceptual_weighting(x,freqs)

def get_zero_crossings_rate(x):
    return librosa.feature.zero_crossing_rate(y=x, pad=False)

def get_zero_crossings(x):
    return np.sum(librosa.feature.zero_crossings(y=x, pad=False))

def get_spectral_centroid(x,sr):
    return librosa.feature.spectral_centroid(y=x, sr=sr)[0]

def get_spectral_bandwidth(x,sr):
    return librosa.feature.spectral_bandwidth(y=x, sr=sr)

def get_spectral_rolloff(x,sr):
    return librosa.feature.spectral_rolloff(y=x+0.01, sr=sr)[0]

def get_chromagraph(x,sr):
    hop_length = 512
    return librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)

def get_rms(x):
    return librosa.feature.rms(y=x)

def get_length(x,sr):
    return list(np.shape(x))[0]/sr

def get_mfccs(x,sr):
    return preprocessing.scale(librosa.feature.mfcc(y=x, sr=sr), axis=1)

def extract_all_features(song_path):
    line = []
    # print(song_path)
    # Get song file name
    song_name = song_path.split("/")[3]
    line.append(song_name)

    # Load song
    x, sr = load_song(song_path)
    
    # Length of audio
    length = get_length(x,sr)
    line.append(length)

    # Chromagram 
    chroma = get_chromagraph(x,sr)
    chroma_avg = calculate_mean(chroma)
    line.append(chroma_avg)
    chroma_var = calculate_var(chroma)
    line.append(chroma_var)

    # RMS
    rms = get_rms(x)
    rms_avg = calculate_mean(rms)
    line.append(rms_avg)
    rms_var = calculate_var(rms)
    line.append(rms_var)

    # Spectral Centroid
    centr = get_spectral_centroid(x,sr)
    centr_avg = calculate_mean(centr)
    line.append(centr_avg)
    centr_var = calculate_var(centr)
    line.append(centr_var)

    # Spectral Bandwidth
    sband = get_spectral_bandwidth(x,sr)
    sband_avg = calculate_mean(sband)
    line.append(sband_avg)
    sband_var = calculate_var(sband)
    line.append(sband_var)

    # Spectral Rolloff
    rolloff = get_spectral_rolloff(x,sr)
    rolloff_avg = calculate_mean(rolloff)
    line.append(rolloff_avg)
    rolloff_var = calculate_var(rolloff)
    line.append(rolloff_var)

    # Zero crossing rate features
    zcr = get_zero_crossings_rate(x)
    zcr_avg = calculate_mean(zcr)
    line.append(zcr_avg)
    zcr_var = calculate_var(zcr)
    line.append(zcr_var)

    # Harmony features
    harmony = get_harmony(x)
    harmony_avg = calculate_mean(harmony)
    line.append(harmony_avg)
    harmony_var = calculate_var(harmony)
    line.append(harmony_var)

    # Perceptual weighting
    perceptr = get_perc_weight(x,sr)
    perceptr_avg = calculate_mean(perceptr)
    line.append(perceptr_avg)
    perceptr_var = calculate_var(perceptr)
    line.append(perceptr_var)

    # Tempo
    tempo = get_tempo(x)
    line.append(tempo)

    # MCSS
    mfccs = get_mfccs(x,sr)
    for mfcc in mfccs:
        line.append(calculate_mean(mfcc))
        line.append(calculate_var(mfcc))

    # Song genre for Labeling
    label = genre_folder.split("/")[2]
    line.append(label)

    # print(line)

    # Add the song features to the feature list
    features.append(line)

for genre_folder in tqdm(glob(os.path.join(path, '*'))):
    # print(genre_folder)
    # print(genre_folder.split("/")[2])
    for song_path in glob(os.path.join(genre_folder,'*.wav')):
        extract_all_features(song_path)
        
data = pd.DataFrame(features,columns=["filename","length","chroma_stft_mean","chroma_stft_var",
    "rms_mean","rms_var","spectral_centroid_mean","spectral_centroid_var",
    "spectral_bandwidth_mean","spectral_bandwidth_var","rolloff_mean",
    "rolloff_var","zero_crossing_rate_mean","zero_crossing_rate_var",
    "harmony_mean","harmony_var","perceptr_mean","perceptr_var","tempo",
    "mfcc1_mean","mfcc1_var","mfcc2_mean","mfcc2_var","mfcc3_mean","mfcc3_var",
    "mfcc4_mean","mfcc4_var","mfcc5_mean","mfcc5_var","mfcc6_mean","mfcc6_var",
    "mfcc7_mean","mfcc7_var","mfcc8_mean",'mfcc8_var',"mfcc9_mean","mfcc9_var",
    "mfcc10_mean","mfcc10_var","mfcc11_mean","mfcc11_var","mfcc12_mean",'mfcc12_var',
    "mfcc13_mean",'mfcc13_var',"mfcc14_mean","mfcc14_var","mfcc15_mean","mfcc15_var",
    "mfcc16_mean","mfcc16_var","mfcc17_mean","mfcc17_var","mfcc18_mean","mfcc18_var",
    "mfcc19_mean",'mfcc19_var',"mfcc20_mean","mfcc20_var","label"])

data.to_csv("features30sec.csv")