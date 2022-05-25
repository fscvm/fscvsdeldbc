import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import ControlDataset, ControlTrainCollate, ControlValidCollate
from model.classifier import Classifier

log_dir = 'logs'
batch_size = 256
lr = 0.001
epochs = 200
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
    model = Classifier(params.n_classes, params.n_feats).cuda()
    print(model)
    print('Number of classifier parameters = %.2fm\n' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses, accs = [], []
        for batch in tqdm(train_loader, total=len(train_dataset)//batch_size):
            x, c = batch['x'].cuda(), batch['c'].cuda()
            lengths = batch['lengths'].cuda()
            z = 2.0 * torch.randn_like(x).cuda().detach()
            model.zero_grad()
            logits = model(x + z, lengths)
            loss, acc = model.compute_loss(logits, c, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc)
            iteration += 1

        losses = np.asarray(losses)
        accs = np.array(accs)
        msg = 'Epoch %d: loss = %.4f | accuracy = %.2f%%\n' % (epoch, np.mean(losses), 100.0*np.mean(accs))
        print(msg)
        with open(f'{log_dir}/cls.log', 'a') as f:
            f.write(msg)
        losses, accs = [], []

        if epoch % save_every > 0:
            continue

        model.eval()
        print('Validation...')
        batch = next(iter(valid_loader))
        x, c, lengths = batch['x'].cuda(), batch['c'].cuda(), batch['lengths'].cuda()
        with torch.no_grad():
            logits = model(x, lengths)
            loss, acc = model.compute_loss(logits, c, lengths)

        msg = 'Validation: loss = %.4f | accuracy = %.2f%%\n' % (loss.item(), 100.0*acc)
        print(msg)
        with open(f'{log_dir}/cls.log', 'a') as f:
            f.write(msg)

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/classifier_{epoch}.pt")
