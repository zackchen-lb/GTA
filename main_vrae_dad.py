import argparse
import os

from exp.exp_vrae_dad import Exp_VRAE_DAD

parser = argparse.ArgumentParser(description='[VRAE] LSTM-VAE for time series modeling')

parser.add_argument('--model', type=str, required=True, default='vrae',help='model of the experiment')

parser.add_argument('--data', type=str, required=True, default='WADI', help='data')
parser.add_argument('--root_path', type=str, default='./data/graph_data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='WADI_14days_downsampled.csv', help='location of the data file')    
parser.add_argument('--features', type=str, default='M', help='features [S, M]')
parser.add_argument('--target', type=str, default='OT', help='target feature')

parser.add_argument('--seq_len', type=int, default=48, help='input series length')
parser.add_argument('--label_len', type=int, default=12, help='help series length')
parser.add_argument('--pred_len', type=int, default=6, help='predict series length')
parser.add_argument('--enc_in', type=int, default=122, help='encoder input size')
parser.add_argument('--enc_hid', type=int, default=256, help='encoder block hidden size')
parser.add_argument('--dec_hid', type=int, default=256, help='decoder block hidden size')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_lat', type=int, default=64, help='dimensionality of latent space')
parser.add_argument('--block', type=str, default='LSTM', help='rnn cell type')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--clip', type=bool, default=False, help='if clipping the gradients')
parser.add_argument('--max_grad_norm', type=int, default=5, help='max gradient normalization coefficient')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'SolarEnergy':{'data':'solar_energy.csv','T':'0','M':[137,137,137],'S':[1,1,1]},
    'WADI':{'data':'WADI_14days_downsampled.csv','T':'1_AIT_005_PV','M':[122,122,122],'S':[1,1,1]},
    'SMAP':{'data':'SMAP','T':0,'M':[25,25,25],'S':[1,1,1]},
    'MSL':{'data':'MSL','T':0,'M':[55,55,55],'S':[1,1,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, _, _ = data_info[args.features]

Exp = Exp_VRAE_DAD

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_eh{}_dh{}_el{}_dl{}_la{}_bl{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.enc_hid, args.dec_hid, args.e_layers, args.d_layers, args.d_lat, args.block, args.des, ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
