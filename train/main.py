import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from rich.prompt import Prompt
from utils import tools
from model import tiny_cnn
from rich import print
import pretty_errors
pretty_errors.activate()


def main():
    # モデル初期化
    #!##################################
    model = tiny_cnn()
    #!##################################
    mode = Prompt.ask("MODE:1.train, 2.test", choices=["1", "2"])
    # 初期セッティング
    device, config, model = tools.init_setting(model=model)
    
    match int(mode):
        case 1:
            # 重みの読み込み
            model = tools.load_weight(model=model, log_dir=config["LOG_DIR"], device=device, pth_name="model.pth")
            # dataloader作成
            train_loader, test_loader, classes = tools.make_loader_cifar10(config=config)
            # 損失関数
            loss_function = nn.CrossEntropyLoss()
            # 最適化関数
            optimizer     = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
            # 学習ループ
            for epoch in range(config["EPOCHS"]):
                running_loss = 0
                for i, data in enumerate(train_loader, 0):
                    # データ取得
                    inputs, labels = data[0].to(device), data[1].to(device)
                    # 勾配初期化
                    optimizer.zero_grad()
                    # 推論
                    outputs = model(inputs)
                    # 損失計算
                    loss = loss_function(outputs, labels)
                    # 逆伝搬
                    loss.backward()
                    # パラメータ更新
                    optimizer.step()
                    # 損失計上
                    running_loss += loss.item()
                
                # エポックごとに損失表示
                print(f'[Epoch {epoch + 1}] loss: {running_loss / len(train_loader):.3f}')
                        
            print("Finished Training\n")
            
            # testデータでテスト
            correct = 0
            total   = 0
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            with torch.no_grad():
                for data in test_loader:
                    # データ格納
                    images, labels = data[0].to(device), data[1].to(device)
                    # 推論
                    outputs = model(images)
                    # 出力確認
                    _, predicted = torch.max(outputs, 1)
                    # 正答率計算
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # ラベルごとの正答率計算
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
            
                    
            print(f"Accuracy of the model on the test images:{100 * correct // total}%\n")
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
                
            

            
            # pth保存
            pth_save_path = "./Python_tools/train/log/model.pth"
            torch.save(model.state_dict(), pth_save_path)
            print(f"pth saved at"+pth_save_path)
            
            # onnx保存
            # 出力path
            onnx_save_path = "./Python_tools/train/outputs/model.onnx"
            # ダミーinput作成
            dummy_input = torch.randn(1, config["INPUT_C"], config["INPUT_H"], config["INPUT_W"]).to(device)
            # 保存
            try:
                torch.onnx.export(
                    model,                      # モデル
                    dummy_input,                # ダミー入力
                    onnx_save_path,                  # 出力パス
                    export_params=True,         # 学習済みパラメータを含める
                    opset_version=21,           # ONNX opset version 
                    do_constant_folding=True,   # 定数畳み込み最適化
                    input_names=['input'],      # 入力名
                    output_names=['output'],    # 出力名
                    dynamic_axes={              # 動的な軸（バッチサイズ）
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            except Exception as e:
                print("Error:----- onnxでの保存に失敗しました。 -----")
            
            
                
            # テスト
            
            return
            
        case 2:
            return
    
    
    
    
    
    
    
    return
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
        