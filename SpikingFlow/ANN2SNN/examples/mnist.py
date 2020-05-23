import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('.')
import SpikingFlow.softbp.neuron as neuron
import SpikingFlow.encoding as encoding
import SpikingFlow.ANN2SNN.transformANN as transformANN
from torch.utils.tensorboard import SummaryWriter
import readline


class Net(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        # 网络结构，简单的双层全连接网络，每一层之后都是IF神经元
        self.fc_seq0 = nn.Flatten()
        self.fc_seq1 = nn.Linear(28 * 28, 1024, bias=False)
        self.fc_seq2 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)
        self.fc_seq3 = nn.Linear(1024, 10, bias=False)
        self.fc_seq4 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.metadata = 'FullyConnect'

    def set_mode(self,mode='SNN'):
        # 利用pytorch的动态图特性，set_mode通过改变对象的属性来改变网络的连接
        if mode == 'SNN':
            self.fc_seq2 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
            self.fc_seq4 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
        elif mode == 'ANN':
            self.fc_seq2 = nn.ReLU()
            self.fc_seq4 = nn.ReLU()

    def forward(self, x):
        x = self.fc_seq0(x)
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_seq3(x)
        x = self.fc_seq4(x)
        return x

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()

def main():
    device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    batch_size = int(input('输入batch_size，例如“64”  '))
    learning_rate = float(input('输入学习率，例如“1e-3”  '))
    T = int(input('输入仿真时长，例如“50”  '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')
    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # 初始化网络
    net = Net().to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    #设置训练方式为ANN
    net.set_mode(mode='ANN')

    for _ in range(train_epoch):
        net.train()
        net.set_mode(mode='ANN')
        for img, label in train_data_loader:
            img = img.to(device)
            optimizer.zero_grad()
            #输出的激活值仿真IFNode的脉冲频率
            out_spikes_analog_frequency = net(img)
            loss = F.cross_entropy(out_spikes_analog_frequency, label.to(device))
            loss.backward()
            optimizer.step()

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            correct_rate = (out_spikes_analog_frequency.max(1)[1] == label.to(device)).float().mean().item()
            writer.add_scalar('train_correct_rate', correct_rate, train_times)
            if train_times % 1024 == 0:
                print(device, dataset_dir, batch_size, learning_rate, T, train_epoch, log_dir)
                print('train_times', train_times, 'train_correct_rate', correct_rate)
            train_times += 1

        net.eval()
        with torch.no_grad():
            # 设置运行模式为ANN
            net.set_mode(mode='ANN')
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                out_spikes_analog_frequency = net(img)
                correct_sum += (out_spikes_analog_frequency.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
            print('ann val_times', train_times, 'val_correct_rate', correct_sum / test_sum)

    net.eval()
    # 最后测试SNN的验证集准确率
    # 设置运行模式为SNN
    net.set_mode(mode='SNN')
    transformANN.normalize_nn(net, train_data_loader, device)
    test_sum = 0
    correct_sum = 0
    for img, label in test_data_loader:
        img = img.to(device)
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(img).float())
            else:
                out_spikes_counter += net(encoder(img).float())

        correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
        test_sum += label.numel()
        net.reset_()
    print('snn val_times', train_times, 'val_correct_rate', correct_sum / test_sum)


if __name__ == '__main__':
    main()
