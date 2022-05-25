import torch
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.filters import chroma as librosa_chroma_fn

frequencies = librosa.fft_frequencies(sr=22050, n_fft=1024)
a_weighting = librosa.A_weighting(frequencies)
weighting = 10**(a_weighting/10)
weighting = torch.from_numpy(weighting).float().unsqueeze(0).unsqueeze(-1).cuda()
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)
mel_basis_inverse = np.linalg.pinv(mel_basis)
mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float().cuda()
chroma_basis = librosa_chroma_fn(22050, 1024, n_chroma=12)
chroma_basis = torch.from_numpy(chroma_basis).float().cuda()

import params
from model.classifier import Classifier
clf = Classifier(params.n_classes, params.n_feats).cuda()
clf.load_state_dict(torch.load('checkpts/classifier.pt'))
clf.train()
print(f'Number of classifier parameters: {clf.nparams}')

from model.pitch import PitchExtractor
pe = PitchExtractor(params.pitch_min, params.pitch_max, params.pitch_mel).cuda()
pe.load_state_dict(torch.load('checkpts/pitch_extractor.pt'))
pe.train()
print(f'Number of pitch extractor parameters: {pe.nparams}')


def power_to_db(power, range_db=80.0):
    pmin = 10**-(range_db / 10.0)
    power = torch.maximum(power, pmin*torch.ones_like(power))
    db = 10.0*torch.log10(power)
    db = torch.maximum(db, -range_db*torch.ones_like(db))
    return db

def compute_loudness_x(log_mel_spectrogram):
    mel_spectrogram = torch.exp(log_mel_spectrogram)
    stftm = torch.matmul(mel_basis_inverse, mel_spectrogram)
    power = (stftm ** 2) * weighting
    loudness = torch.mean(power, 1, keepdim=True)
    loudness = power_to_db(loudness)
    return loudness

def compute_loudness_z(log_mel_spectrogram, feat_mean, feat_std):
    mel_spectrogram = torch.exp(log_mel_spectrogram * feat_std + feat_mean)
    stftm = torch.matmul(mel_basis_inverse, mel_spectrogram)
    power = (stftm ** 2) * weighting
    loudness = torch.mean(power, 1, keepdim=True)
    loudness = power_to_db(loudness)
    return loudness

def compute_chroma_x(log_mel_spectrogram):
    mel_spectrogram = torch.exp(log_mel_spectrogram)
    stftm = torch.matmul(mel_basis_inverse, mel_spectrogram)
    power = (stftm ** 2) * weighting
    chroma = torch.matmul(chroma_basis, power)
    chroma /= torch.max(chroma, 1, keepdim=True)[0]
    return chroma

def compute_chroma_z(log_mel_spectrogram, feat_mean, feat_std):
    mel_spectrogram = torch.exp(log_mel_spectrogram * feat_std + feat_mean)
    stftm = torch.matmul(mel_basis_inverse, mel_spectrogram)
    power = (stftm ** 2) * weighting
    chroma = torch.matmul(chroma_basis, power)
    chroma /= torch.max(chroma, 1, keepdim=True)[0]
    return chroma

def compute_prob_x(x):
    lengths = torch.LongTensor([x.shape[-1]]).cuda()
    return torch.nn.functional.softmax(clf(x, lengths), dim=1)

def compute_prob_z(x, feat_mean, feat_std):
    x = x * feat_std + feat_mean
    lengths = torch.LongTensor([x.shape[-1]]).cuda()
    return torch.nn.functional.softmax(clf(x, lengths), dim=1)

def compute_pitch_x(x):
    lengths = torch.LongTensor([x.shape[-1]]).cuda()
    return pe(x, lengths)

def compute_pitch_z(x, feat_mean, feat_std):
    x = x * feat_std + feat_mean
    lengths = torch.LongTensor([x.shape[-1]]).cuda()
    return pe(x, lengths)
