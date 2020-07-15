# インポート
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=1000, shuffle=True)

'''
for batch_idx, (data, target) in enumerate(train_loader):
    print(data)
    print(target)
    exit()
'''

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
    print('%d epoch, loss=%.5f' %
                      (epoch+1, loss.item()))


model.eval()    # 評価モード
correct = 0    # 予測成功数のカウント
with torch.no_grad():     # 勾配計算の無効化
    for data, target in test_loader:    # テストデータに対する繰り返し処理
        data, target = data.to(device), target.to(device)  # データをターゲットデバイスへ移動
        output = model(data)   # フォワード・パス
        pred = output.max(1, keepdim=True)[1] # 最も高い確率のインデックス取得
        correct += pred.eq(target.view_as(pred)).sum().item()    # 予測が正しければカウント

print("Test set: Accuracy: "+str(correct/10000*100)+"%")