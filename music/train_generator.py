import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from model import MusicGenerator
from data import MusicDataset, MusicBatchCollate
from utils import save_audio

import json
import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

train_filelist = params.train_filelist
exc_filelist = params.exc_filelist

n_feats = params.n_feats
n_classes = params.n_classes

base_dim = params.base_dim
class_dim = params.class_dim
beta_min = params.beta_min
beta_max = params.beta_max

log_dir = params.log_dir
batch_size = params.batch_size
train_length = params.train_length
lr = params.lr
epochs = params.epochs
save_every = params.save_every

random_seed = params.seed

with open('checkpts/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
hifigan = HiFiGAN(h).cuda()
hifigan.load_state_dict(torch.load('checkpts/hifigan-music.pt')['generator'])
hifigan.eval()
hifigan.remove_weight_norm()


def get_test_batch():
    data_path = '../data/Music/'
    x_test_0 = [np.load(data_path + 'mels/piano/0sDleZkIK-w_116.npy')]*3
    x_test_1 = [np.load(data_path + 'mels/flute/6GwfuWhOOdY_37.npy')]*3
    x_test_2 = [np.load(data_path + 'mels/harpsichord/5B4eEcvBIek_114.npy')]*3
    x_test_3 = [np.load(data_path + 'mels/string/npQJP_nF7NI_125.npy')]*3
    x_test = np.stack(x_test_0 + x_test_1 + x_test_2 + x_test_3, axis=0)
    lengths_test = torch.LongTensor([x_test.shape[-1]]*12).cuda()
    x_test = torch.from_numpy(x_test).float().cuda()
    c_source_test = torch.LongTensor([0]*3 + [1]*3 + [2]*3 + [3]*3).cuda()
    c_target_test = torch.LongTensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]).cuda()
    with torch.no_grad():
        audio = hifigan.forward(x_test)
    save_audio(f'{log_dir}/piano.wav', 22050, audio[0, :, :])
    save_audio(f'{log_dir}/flute.wav', 22050, audio[3, :, :])
    save_audio(f'{log_dir}/harpsichord.wav', 22050, audio[6, :, :])
    save_audio(f'{log_dir}/string.wav', 22050, audio[9, :, :])
    return x_test, lengths_test, c_source_test, c_target_test

def save_test_results(audio):
    save_audio(f'{log_dir}/piano2flute.wav', 22050, audio[0, :, :])
    save_audio(f'{log_dir}/piano2harpsichord.wav', 22050, audio[1, :, :])
    save_audio(f'{log_dir}/piano2string.wav', 22050, audio[2, :, :])
    save_audio(f'{log_dir}/flute2piano.wav', 22050, audio[3, :, :])
    save_audio(f'{log_dir}/flute2harpsichord.wav', 22050, audio[4, :, :])
    save_audio(f'{log_dir}/flute2string.wav', 22050, audio[5, :, :])
    save_audio(f'{log_dir}/harpsichord2piano.wav', 22050, audio[6, :, :])
    save_audio(f'{log_dir}/harpsichord2flute.wav', 22050, audio[7, :, :])
    save_audio(f'{log_dir}/harpsichord2string.wav', 22050, audio[8, :, :])
    save_audio(f'{log_dir}/string2piano.wav', 22050, audio[9, :, :])
    save_audio(f'{log_dir}/string2flute.wav', 22050, audio[10, :, :])
    save_audio(f'{log_dir}/string2harpsichord.wav', 22050, audio[11, :, :])
    return


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing data loaders...')
    train_dataset = MusicDataset(train_filelist, exc_filelist)
    batch_collate = MusicBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True, 
                        num_workers=4, shuffle=True)
    feat_mean = train_dataset.feat_mean
    feat_std = train_dataset.feat_std

    print('Preparing test batch...')
    x_test, lengths_test, c_source_test, c_target_test = get_test_batch()

    print('Initializing model...')
    model = MusicGenerator(n_feats, n_classes, base_dim, class_dim, 
                           beta_min, beta_max, feat_mean, feat_std).cuda()
    print('Music Generator:')
    print(model)
    print('Number of parameters = %.2fm\n' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(loader, total=len(train_dataset)//batch_size):
            x, lengths, c = batch['x'].cuda(), batch['lengths'].cuda(), batch['c'].cuda()
            model.zero_grad()
            loss = model.compute_loss(x, lengths, c)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                       max_norm=1)
            optimizer.step()

            losses.append(loss.item())
            iteration += 1

        msg = 'Epoch %d: loss = %.3f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % save_every > 0:
            continue

        model.eval()
        print('Conversion...\n')
        with torch.no_grad():
            y_test = model.convert(x_test, lengths_test, c_source_test, 
                                   c_target_test, n_timesteps=50)
            audio = hifigan.forward(y_test)
            save_test_results(audio)

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/music_generator_{epoch}.pt")
