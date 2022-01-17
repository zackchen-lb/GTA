import torch
from models.gta import GraphTemporalEmbedding

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(32, 96, 122)
    model = GraphTemporalEmbedding(122, 96, 3)
    y = model(x)
    print(y.size())
    # model = AdaGraphSage(num_nodes=10, seq_len=96, label_len=48, out_len=24)
    # model = model.double().to(device)
    # x = torch.randn(32, 96, 10, requires_grad=True).double().to(device)
    # y = torch.randn(32, 48, 10, requires_grad=True).double().to(device)
    # # print(out.size())
    # out = model(x, y, None, None)
    # print(out.size())