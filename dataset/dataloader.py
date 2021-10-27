import torch
from dataset.common import load_info
from dataset.tdmatch import TDMatchDataset as TDMatchData
from dataset.kitti import KITTIDataset as KITTIData
from functools import partial
from model.KPConv.preprocessing import collate_fn_descriptor, calibrate_neighbors


def get_dataset(config):
    '''
    Make pytorch dataset for train, val and benchmark dataset
    :param config: configuration
    :return: train_set: training dataset
             val_set: val dataset
             benchmark_set: benchmark dataset
    '''

    if config.dataset == 'tdmatch':
        info_train = load_info(config.train_info)
        info_val = load_info(config.val_info)
        info_benchmark = load_info(f'configs/tdmatch/{config.benchmark}.pkl')
        train_set = TDMatchData(info_train, config, data_augmentation=True)
        val_set = TDMatchData(info_val, config, data_augmentation=False)
        benchmark_set = TDMatchData(info_benchmark, config, data_augmentation=False)
    elif config.dataset == 'kitti':
        train_set = KITTIData(config, 'train', data_augmentation=True)
        val_set = KITTIData(config, 'val', data_augmentation=False)
        benchmark_set = KITTIData(config, 'test', data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set


def get_dataloader(dataset, batch_size=1, num_workers=4, shuffle=True, neighborhood_limits=None):
    '''
    Get the pytorch dataloader for specific dataset
    :param dataset: Pytorch dataset
    :param batch_size: The size of batch data
    :param num_workers: Multi-threads data loader
    :param shuffle: Whether to shuffle the dataset
    :return: Pytorch dataloader, neighborhood limits for KPConv
    '''
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
    print('neighborhood: ', neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits

