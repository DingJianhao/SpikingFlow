import torch
import torch.nn.functional as F


def normalize_nn(net, train_loader,device):
    # method from paper 《Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing》
    assert('bias' not in net.metadata)
    assert('BatchNorm' not in net.metadata)

    activation_max = dict()
    weight_max = dict()

    for n,m in net.named_modules():
        if m.__class__.__name__ in ['Linear','Conv2d']:
            activation_max[n] = -1e5
            weight_max[n] = torch.max(F.relu(m.weight))
    correct_sum = 0
    test_sum = 0
    net.set_mode('ANN')
    with torch.no_grad():
        net.eval()
        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            x = img
            for n, m in net.named_modules():
                if m.__class__.__name__ in ['Linear', 'Conv2d','Flatten']:  # forward
                    x = m(x)
                    if m.__class__.__name__ in ['Linear', 'Conv2d']:
                        activation_max[n] = max(activation_max[n], torch.max(x))
            correct_sum += (x.max(1)[1] == label.to(device)).float().mean().item()
            test_sum += label.numel()

    previous_factor = 1.0
    for n, m in net.named_modules():
        if m.__class__.__name__ in ['Linear', 'Conv2d']:  # forward
            scale_factor = max(weight_max[n], activation_max[n])
            applied_inv_factor = (scale_factor / previous_factor).item()
            m.weight.data = m.weight.data / applied_inv_factor
            previous_factor = applied_inv_factor
