# random seed
seed = 37

# data path
train_filelist = 'filelists/train.txt'
valid_filelist = 'filelists/valid.txt'
exc_filelist = 'filelists/exceptions.txt'

# data parameters
n_feats = 80
n_classes = 4
pitch_min = 3.5
pitch_max = 8.0
pitch_mel = 40

# diffusion parameters
base_dim = 96
class_dim = 64
beta_min = 0.05
beta_max = 20.0

# training parameters
log_dir = 'logs'
batch_size = 16
train_length = 256
lr = 0.0001
epochs = 1850
save_every = 25
