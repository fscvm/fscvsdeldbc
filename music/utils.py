import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        lines = [line.strip().split(split_char) for line in f]
    return lines


def save_mel(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.detach().cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))
    return

def save_pitch(tensor1, tensor2, savepath):
    plt.figure()
    plt.plot(tensor1.squeeze().detach().cpu().numpy())
    plt.plot(tensor2.squeeze().detach().cpu().numpy())
    plt.savefig(savepath)
    plt.close()
    return
