# data parameters
n_mels = 80
sampling_rate = 22050
n_fft = 1024
hop_size = 256

# "average voice" encoder parameters
channels = 192
filters = 768
layers = 6
kernel = 3
dropout = 0.1
heads = 2
window_size = 4
enc_dim = 128

# diffusion-based decoder parameters
dec_dim = 256
spk_dim = 128
use_ref_t = True
beta_min = 0.05
beta_max = 20.0

# training parameters
seed = 37
test_size = 1
train_frames = 128
