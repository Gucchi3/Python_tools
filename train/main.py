import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from rich.prompt import Prompt
from utils import tools




def main():
    mode = Prompt.ask("MODE:1.train, 2.test", choices=["1", "2"])
    # 初期セッティング
    device, config, model = tools.init_setting(model=model)
    
    match int(mode):
        case 1:
            # 重みの読み込み
            model = tools.load_weight(model=model, log_dir=config["LOG_DIR"], device=device, pth_name="best.pth")
            # dataloader作成
            train_loader, test_loader, classes = tools.make_loader_cifar10(config=config)
            # 損失関数
            loss_function = nn.CrossEntropyLoss()
            # 最適化関数
            optimizer     = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
            # 学習ループ
            # テスト
            
            return
            
        case 2:
            return
    
    
    
    
    
    
    
    return
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
        