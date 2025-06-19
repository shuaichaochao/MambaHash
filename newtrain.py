from utils.tools import *
# from model.network import *

from torch.cuda.amp import autocast,GradScaler
import torch.nn as nn
from functools import partial
import torch
import torch.optim as optim
import time
# from apex import amp
from loguru import logger
# from model.HybridHash import HybridHash
# from model.vit import VisionTransformer, CONFIGS
from model.MambaHash import GroupMamba
from model.network import ResNet
from ptflops import get_model_complexity_info
# from apex import amp
from utils.Hash_loss import DPSHLoss
torch.multiprocessing.set_sharing_strategy('file_system')
import random, os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config():
    config = {
        "alpha": 0.01,
        "lamada": 0,
        
        # "gamma":0.4,

        "optimizer":"RMSprop",
        # "optimizer":"SGD",
        # "optimizer": "Adam",
        # "optimizer": {"type": optim.AdamW, "optim_params": {"lr": 1e-4, "weight_decay": 1e-7}, "lr_type": "step"},
        "lr": 2e-5,
        "weight_decay":1e-7,
        "momen":0.9,
        "lr_step":"10",
        "info": "[MambaHash]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        # "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 8,
        "test_map": 1,
        "save_path": "save/HashNet",
        "device": torch.device("cuda:3"),
        "bit_list": [16, 32, 48, 64],
        # "bit_list": [64],
        "pretrained_dir": "checkpoint/groupmamba_small_ema.pth",
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 3,
        "model_type": "ViT-L_16",
        "top_img": 10,
    }
    config = config_dataset(config)
    return config


def train_val(config, bit):
    seed_everything(2025)
    device = config["device"]
    # Prepare model
    # configs = CONFIGS[config["model_type"]]
    # net = config["net"](bit).to(device)


    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["bit"] = bit


    net = GroupMamba( hash_bit=config["bit"],
                num_classes=config["n_class"],
                stem_hidden_dim = 64,
                # embed_dims = [96, 192, 424, 512],
                embed_dims = [64, 128, 348, 512],
                mlp_ratios = [8, 8, 4, 4],
                norm_layer = partial(nn.LayerNorm, eps=1e-6), 
                depths = [3, 4, 16, 3],
                k_size = [3,3,5,5]
                )
    

    if config["pretrained_dir"] is not None:
        logger.info('Loading:', config["pretrained_dir"])
        state_dict = torch.load(config["pretrained_dir"])
        net.load_state_dict(state_dict, strict=False)
        logger.info('Pretrain weights loaded.')


    net = net.to(device)

     # 计算模型计算力和参数量（Statistical model calculation and number of parameters）
    flops, num_params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # logger.info("{}".format(config))
    logger.info("Total Parameter: \t%s" % num_params)
    logger.info("Total Flops: \t%s" % flops)



    if config["optimizer"] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momen"])
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif config["optimizer"] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config["lr_step"], gamma=0.1)


    #原声自带apex训练
    scaler = GradScaler()

    # apex加速训练
    # help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
    # See details at https://nvidia.github.io/apex/amp.html"
    # net, optimizer = amp.initialize(models=net,
    #                                   optimizers=optimizer,
    #                                   opt_level='O1')
    # amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
  
    criterion = DPSHLoss(config, bit)

    Best_mAP = 0.0

    net.train()
    optimizer.zero_grad()


    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")


        train_loss = 0
        allSimilarloss = 0
        allquantizationloss = 0
        for idx, (image, label, ind) in enumerate(train_loader):

            image = image.to(device)
            label = label.to(device)

            with autocast():
                u = net(image)
                loss, Similarloss, quantization_loss = criterion(u, label.float(), ind, config)

            train_loss += loss.item()
            allSimilarloss += Similarloss.item()
            allquantizationloss += quantization_loss.item()
            # apex加速训练
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            # loss.backward()
            
            # optimizer.step()

            #原生自带apex
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_loss / len(train_loader)
        allSimilarloss = allSimilarloss / len(train_loader)
        allquantizationloss = allquantizationloss / len(train_loader)

        scheduler.step()

        logger.info("\b\b\b\b\b\b\b loss:%.5f, allSimilarloss: %.5f, allquantizationloss: %.5f"  % (train_loss, allSimilarloss, allquantizationloss))
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, net, epoch)


def test(config, Best_Accuracy, test_loader, net):
    device = config["device"]
    logger.info("Calculate test classification......")
    net.eval()
    for img, cls, _ in tqdm(test_loader, ncols=60):
        _, predicted = torch.max(net(img.to(device)).data, dim=1)
        _, true_cls = torch.max(cls.to(device), dim=1)
        Best_Accuracy += (predicted == true_cls).sum().item()

        net.train()

    return Best_Accuracy



if __name__ == "__main__":
    # 原本自己的
    config = get_config()

    # 建立日志文件（Create log file）
    # logger.add('logs/{time}' + config["info"] + '_' + config["dataset"] + ' alpha '+str(config["alpha"]) + '.log', rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        # config["pr_curve_path"] = f"log/alexnet/HashNet_{config['dataset']}_{bit}.json"
        train_val(config, bit)
