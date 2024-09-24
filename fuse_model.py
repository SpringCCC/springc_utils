import torch
import torch.nn as nn

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    a = bn.weight.div(torch.sqrt(bn.eps + bn.running_var)) 
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    fusedconv.weight.copy_((w_conv * a.reshape(-1, 1)).view(fusedconv.weight.shape))
    tmp = (conv_bias-bn.running_mean)*a+bn.bias
    fusedconv.bias.copy_(tmp)
    return fusedconv