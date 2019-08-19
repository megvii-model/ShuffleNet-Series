# -*- coding: utf-8 -*-
# @Time    : 2019-07-30 23:26 
# @Author  : Yi Zou
# @File    : flops_counter.py 
# @Software: PyCharm


import torch
import torch.nn as nn


mode = 'MADDs'


def count_conv2d(m, x, y):
    assert mode in ['MMULs', 'MADDs']
    x = x[0]

    _, in_channels, in_h, in_w = x.size()
    _, out_channels, out_h, out_w = y.size()
    kernel_h, kernel_w = m.kernel_size

    kernel_ops = kernel_h * kernel_w * in_channels // m.groups
    bias_ops = 1 if m.bias is not None else 0
    if mode == 'MMULs':
        ops_per_element = kernel_ops
    else:
        ops_per_element = kernel_ops + bias_ops

    out_elements = out_channels * out_h * out_w
    total_ops = int(out_elements * ops_per_element)
    m.total_ops = torch.Tensor([total_ops, ])


def count_linear(m, x, y):
    assert mode in ['MMULs', 'MADDs']
    x = x[0]

    mul_ops = m.in_features
    bias_ops = 1 if m.bias is not None else 0
    if mode == 'MMULs':
        ops_per_element = mul_ops
    else:
        ops_per_element = mul_ops + bias_ops
    out_elements = m.out_features
    total_ops = int(out_elements * ops_per_element)
    m.total_ops = torch.Tensor([total_ops, ])


register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.Linear: count_linear,
}


def profile(model: nn.Module, input_size: tuple):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if type(m) in register_hooks:
            fn = register_hooks[type(m)]
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    model.eval().to('cpu')
    model.apply(add_hooks)
    with torch.no_grad():
        model(torch.zeros(input_size).to('cpu'))

    model_ops = 0
    model_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        model_ops += m.total_ops
        model_params += m.total_params

    model_ops = model_ops.item()
    model_params = model_params.item()

    model.train().to('cpu')
    for handler in handler_collection:
        handler.remove()

    return int(model_ops), int(model_params)


def print_profile(model: nn.Module, input_size=(1, 3, 224, 224), mode='SIMPLE'):
    assert mode in ['SIMPLE', 'ALL']
    model_ops, model_params = profile(model, input_size)
    if mode == 'SIMPLE':
        print('FLOPs: ' + str(model_ops // 1e4 / 100.) + 'M', 'PARAMs: ' + str(model_params // 1e4 / 100.) + 'M')
    else:
        for name, m in model.named_modules():
            if len(list(m.children())) == 0 and m.total_ops != 0:
                ops, params = str(m.total_ops.item() // 1e4 / 100.), str(m.total_params.item() // 1e4 / 100.)
                print('Name: %35s    FLOPs: %10sM    PARAMs: %10sM' % (name, str(ops), str(params)))
        print('FLOPs: ' + str(model_ops), 'PARAMs: ' + str(model_params))














