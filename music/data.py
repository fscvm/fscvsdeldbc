import random
import numpy as np
import torch

from utils import parse_filelist
from model.utils import fix_len_compatibility
from params import seed as random_seed
from params import n_feats, train_length


class ControlDataset(torch.utils.data.Dataset):
    def __init__(self, filelist, exceptions):
        self.exc_idx = [x[0] for x in parse_filelist(exceptions)]
        self.feats_and_idx = [x for x in parse_filelist(filelist)
                              if x[0] not in self.exc_idx]
        random.seed(random_seed)
        random.shuffle(self.feats_and_idx)

    def get_tuple(self, feat_and_id):
        feat_path = feat_and_id[0]
        instrument_id = feat_and_id[1]
        pitch_path = feat_path.replace('/mels/', '/pitch/')
        feat = self.get_feat(feat_path)
        instrument_id = self.get_instrument(instrument_id)
        pitch = self.get_pitch(pitch_path)
        return (feat, instrument_id, pitch)

    def get_feat(self, feat_path):
        feat = np.load(feat_path)
        return torch.from_numpy(feat).float()

    def get_instrument(self, instrument_id):
        return torch.LongTensor([int(instrument_id)])

    def get_pitch(self, pitch_path):
        pitch = np.load(pitch_path)
        return torch.from_numpy(pitch).float().unsqueeze(0)

    def __getitem__(self, index):
        feat, instrument_id, pitch = self.get_tuple(self.feats_and_idx[index])
        item = {'x': feat, 'c': instrument_id, 'p': pitch}
        return item

    def __len__(self):
        return len(self.feats_and_idx)


class ControlTrainCollate(object):
    def __call__(self, batch):
        B = len(batch)

        n_feats = batch[0]['x'].shape[0]
        x = torch.zeros((B, n_feats, train_length), dtype=torch.float32)
        p = torch.zeros((B, 1, train_length), dtype=torch.float32)

        max_starts = [max(item['x'].shape[-1] - train_length, 0)
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        lengths = []
        c = []

        for i, item in enumerate(batch):
            x_item, c_item, p_item = item['x'], item['c'], item['p']
            if x_item.shape[-1] < train_length:
                length = x_item.shape[-1]
            else:
                length = train_length
            lengths.append(length)
            c.append(c_item)
            x[i, :, :length] = x_item[:, starts[i]:starts[i] + length]
            p[i, :, :length] = p_item[:, starts[i]:starts[i] + length]

        lengths = torch.LongTensor(lengths)
        c = torch.cat(c, 0)
        return {'x': x, 'lengths': lengths, 'c': c, 'p': p}

    
class ControlValidCollate(object):
    def __call__(self, batch):
        B = len(batch)

        n_feats = batch[0]['x'].shape[0]
        max_length = max([item['x'].shape[-1] for item in batch])
        x = torch.zeros((B, n_feats, max_length), dtype=torch.float32)
        p = torch.zeros((B, 1, max_length), dtype=torch.float32)
        lengths = []
        c = []

        for i, item in enumerate(batch):
            x_item, c_item, p_item = item['x'], item['c'], item['p']
            length = x_item.shape[-1]
            lengths.append(length)
            c.append(c_item)
            x[i, :, :length] = x_item
            p[i, :, :length] = p_item

        lengths = torch.LongTensor(lengths)
        c = torch.cat(c, 0)
        return {'x': x, 'lengths': lengths, 'c': c, 'p': p}


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, filelist, exceptions):
        self.exc_idx = [x[0] for x in parse_filelist(exceptions)]
        self.feats_and_idx = [x for x in parse_filelist(filelist)
                              if x[0] not in self.exc_idx]
        print('%d training files found.' % len(self.feats_and_idx))
        print('Calculating stats...')
        feats = []
        for feat_and_id in self.feats_and_idx:
            feat_path = feat_and_id[0]
            feat = np.load(feat_path)
            feats.append(feat)
        feats = np.concatenate(feats, axis=-1)
        self.feat_mean = np.mean(feats)
        self.feat_std = np.std(feats)
        print('Mean = %.2f | Std = %.2f' % (self.feat_mean, self.feat_std))
        random.seed(random_seed)
        random.shuffle(self.feats_and_idx)

    def get_tuple(self, feat_and_id):
        feat_path = feat_and_id[0]
        instrument_id = feat_and_id[1]
        feat = self.get_feat(feat_path)
        instrument_id = self.get_instrument(instrument_id)
        return (feat, instrument_id)

    def get_feat(self, feat_path):
        feat = np.load(feat_path)
        feat = (feat - self.feat_mean) / self.feat_std
        return torch.from_numpy(feat).float()

    def get_instrument(self, instrument_id):
        return torch.LongTensor([int(instrument_id)])

    def __getitem__(self, index):
        feat, instrument_id = self.get_tuple(self.feats_and_idx[index])
        item = {'x': feat, 'c': instrument_id}
        return item

    def __len__(self):
        return len(self.feats_and_idx)


class MusicBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        max_length = fix_len_compatibility(train_length)

        x = torch.zeros((B, n_feats, max_length), dtype=torch.float32)

        max_starts = [max(item['x'].shape[-1] - max_length, 0)
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        lengths = []
        c = []

        for i, item in enumerate(batch):
            x_item, c_item = item['x'], item['c']
            if x_item.shape[-1] < max_length:
                length = x_item.shape[-1]
            else:
                length = max_length
            lengths.append(length)
            c.append(c_item)
            x[i, :, :length] = x_item[:, starts[i]:starts[i] + length]

        lengths = torch.LongTensor(lengths)
        c = torch.cat(c, 0)
        return {'x': x, 'lengths': lengths, 'c': c}
