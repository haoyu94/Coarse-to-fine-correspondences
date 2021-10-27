import os, argparse, json, shutil
import torch
from torch import optim
from lib.utils import setup_seed
from configs.config_utils import load_config
from easydict import EasyDict as edict
from lib.loss import MetricLossOT as MetricLoss
from lib.tester import get_trainer
from dataset.dataloader import get_dataset, get_dataloader
from model.KPConv.architectures import architectures
from model.Models.roughmatching import RoughMatchingModel

setup_seed(0)


def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)

    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']
    config['visual_dir'] = 'snapshot/%s/visualization' % config['exp_dir']
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.visual_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # backup the files
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r dataset {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    shutil.copy2('main.py', config.snapshot_dir)

    # model initialization
    config.architecture = architectures[config.arch]
    print(config.architecture)
    config.model = RoughMatchingModel(config)

    print(config.model)
    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )

    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_dataset(config)
    config.train_loader, neighborhood_limits = get_dataloader(train_set,
                                                              batch_size=config.batch_size,
                                                              num_workers=config.num_workers,
                                                              shuffle=True)

    config.val_loader, _ = get_dataloader(val_set,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       neighborhood_limits=neighborhood_limits)

    config.test_loader, _ = get_dataloader(benchmark_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        neighborhood_limits=neighborhood_limits)
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)

    trainer = get_trainer(config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'val':
        trainer.eval()
    else:
        trainer.test()


if __name__ == '__main__':
    main()
