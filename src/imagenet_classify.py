from download_imagenet import data_download as download
from pytorch_cnn import Net as Net
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

file_name = '../data/'+download()
traindir = os.path.join(file_name, 'train')     # /train/ を指定されたパスに追加
testdir = os.path.join(file_name, 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])     # 正規化定数

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),      # 画像をサイズ224に切り出しもしくはリサイズ
            transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
            transforms.ToTensor(),
            normalize,
        ]))

train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True)

test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),      # 画像をサイズ224に切り出しもしくはリサイズ
            transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
            transforms.ToTensor(),
            normalize,
        ]))

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True)

model = Net()

device = torch.device("cpu")  # マシンのCPUを使用する
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)   # 最適化ルーチンの作成
model.train()     # 訓練モードの設定
for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader): # 分割データの繰り返し処理
        data, target = data.to(device), target.to(device)  # データをターゲットデバイスに移動
        optimizer.zero_grad()  # 勾配を0に設定
        output = model.forward(data)   # フォワード・パス
        loss = F.nll_loss(output, target)   # 誤差計算
        loss.backward()  # 逆伝播計算
        optimizer.step()   # 次ステップへの準備
        # 100回に1回進捗を表示（なくてもよい）
        if batch_idx%100==0:
            print('%03d epoch, %05d, loss=%.5f' %
                    (epoch, batch_idx, loss.item()))


model.eval()    # 評価モード
correct = 0    # 予測成功数のカウント
with torch.no_grad():     # 勾配計算の無効化
    for data, target in test_loader:    # テストデータに対する繰り返し処理
        data, target = data.to(device), target.to(device)  # データをターゲットデバイスへ移動
        output = model(data)   # フォワード・パス
        pred = output.max(1, keepdim=True)[1] # 最も高い確率のインデックス取得
        correct += pred.eq(target.view_as(pred)).sum().item()    # 予測が正しければカウント

print("Test set: Accuracy: "+str(correct/20000*100)+"%")