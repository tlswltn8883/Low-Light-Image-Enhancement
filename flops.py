from thop import profile
import torch
import time
from model import RawFormer

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = RawFormer().to(device)
    input = torch.randn(1, 3, 256, 256).to(device)
    with torch.cuda.device(device):
        torch.cuda.synchronize()
        time_start = time.time()
        _ = model(input)
        torch.cuda.synchronize()
        time_end = time.time()

    time_sum = time_end - time_start
    print(f"Time: {time_sum}")
    flops, params = profile(model, inputs=(input,))
    print('flops:{}G params:{}M'.format(flops / 1e9, params / 1e6))