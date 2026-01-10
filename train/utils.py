import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import json
import os









class tools():
    @staticmethod
    def init_setting(model, seed=11):
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # modelをcudaもしくはcpuに配置
        model = model.to(device)
        # 乱数初期化
        random.seed(seed)
        # config.jsonの読み込み -> outputs, logフォルダを読み込み -> なければ作成
        with open("../config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        # フォルダパス読み込み
        log_dir     = config["LOG_DIR"]
        outputs_dir = config["OUTPUTS_DIR"]
        # log, outputs フォルダを作成
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        # return
        return device, config, model
    
    @staticmethod
    def load_weight(model, log_dir, device, pth_name="best.pth"):
        # best.pthのパスを取得
        best_weight_path = os.path.join(log_dir,pth_name)
        if os.path.exists(best_weight_path):
            try:
                model.load_state_dict(torch.load(best_weight_path, map_location=device))
                print("Success load_weight\n")
            except Exception as e:
                print("Error: ----- tools.load_weight -----\n")
                return
        # return
        return model
    
    @staticmethod
    def make_loader_cifar10(config):
        transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
        
        train_set    = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"])
        
        test_set     = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)
        test_loader  = torch.utils.data.DataLoader(test_set, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
        
        classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        return train_loader, test_loader, classes
    
    
    