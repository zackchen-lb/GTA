import argparse
import os

from exp.exp_gta_dad import Exp_GTA_DAD

parser = argparse.ArgumentParser(description='[GTA] GTA for DAD')

parser.add_argument('--model', type=str, required=True, default='gta',help='model of the experiment')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')    
parser.add_argument('--features', type=str, default='M', help='features [S, M]')
parser.add_argument('--target', type=str, default='OT', help='target feature')

parser.add_argument('--seq_len', type=int, default=60, help='input series length')
parser.add_argument('--label_len', type=int, default=30, help='help series length')
parser.add_argument('--pred_len', type=int, default=24, help='predict series length')
parser.add_argument('--num_nodes', type=int, default=7, help='encoder input size')
parser.add_argument('--num_levels', type=int, default=3, help='number of dilated levels for graph embedding')
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='prob sparse factor')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1]},
    'SolarEnergy':{'data':'solar_energy.csv','T':'0','M':[137,137,137],'S':[1,1,1]},
    'WADI':{'data':'WADI_14days_downsampled.csv','T':'1_LS_001_AL','M':112,'S':1},
    'SMAP':{'data':'SMAP','T':0,'M':25,'S':1},
    'MSL':{'data':'MSL','T':0,'M':55,'S':1},
    'SWaT':{'data':'SWaT','T':'FIT_101','M':51,'S':1}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.num_nodes = data_info[args.features]

Exp = Exp_GTA_DAD

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_nl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len, args.num_levels,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.embed, args.des, ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
