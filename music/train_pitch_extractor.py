import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import ControlDataset, ControlTrainCollate, ControlValidCollate
from model.pitch import PitchExtractor
from utils import save_pitch

log_dir = 'logs'
batch_size = 256
lr = 0.001
epochs = 1000
save_every = 10

train_filelist = params.train_filelist
valid_filelist = params.valid_filelist
exc_filelist = params.exc_filelist
random_seed = params.seed


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing data loaders...')
    train_dataset = ControlDataset(train_filelist, exc_filelist)
    valid_dataset = ControlDataset(valid_filelist, exc_filelist)
    train_collate = ControlTrainCollate()
    valid_collate = ControlValidCollate()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              collate_fn=train_collate, drop_last=True, 
                              num_workers=4, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset),
                              collate_fn=valid_collate, drop_last=False, 
                              shuffle=False)
    print('%d files for training, %d files for validation.' % (len(train_dataset), len(valid_dataset)))

    print('Initializing model...')
    model = PitchExtractor(params.pitch_min, params.pitch_max, params.pitch_mel).cuda()
    print(model)
    print('Number of pitch extractor parameters = %.2fm\n' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_dataset)//batch_size):
            x, p = batch['x'].cuda(), batch['p'].cuda()
            x = x[:, :params.pitch_mel, :]
            lengths = batch['lengths'].cuda()
            model.zero_grad()
            p_predicted = model(x, lengths)
            loss = model.compute_loss(p_predicted, p, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            iteration += 1

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir}/pitch.log', 'a') as f:
            f.write(msg)
        losses = []

        if epoch % save_every > 0:
            continue

        model.eval()
        print('Validation...')
        batch = next(iter(valid_loader))
        x, p, lengths = batch['x'].cuda(), batch['p'].cuda(), batch['lengths'].cuda()
        x = x[:, :params.pitch_mel, :]
        with torch.no_grad():
            p_predicted = model(x, lengths)
            loss = model.compute_loss(p_predicted, p, lengths)
        save_pitch(p[0, :, :], p_predicted[0, :, :], f'{log_dir}/pitch.png')

        msg = 'Validation: loss = %.4f\n' % loss.item()
        print(msg)
        with open(f'{log_dir}/pitch.log', 'a') as f:
            f.write(msg)

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/pitch_extractor_{epoch}.pt")
