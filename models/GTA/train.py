import torch
from torch import nn
from torch.utils.data import DataLoader

from gragod.utils import load_params, load_training_data
from gragod.tools import dotdict
from models.GTA.exp.exp_informer import Exp_Informer
from gragod.metrics import mae, mse, rmse



DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"


def main(params):
    # Load data
    X_train, X_val, X_test, *_ = load_training_data(
        DATA_PATH, normalize=False, clean=True
    )

    args = dotdict()

    args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'custom' # data
    args.root_path = './' # root path of data file
    args.data_path = 'TELCO_data.csv' # data file
    args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'target' # target feature in S or MS task
    args.freq = '5m' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = './informer_checkpoints' # location of model checkpoints

    args.seq_len = 96 # input sequence length of Informer encoder
    args.label_len = 48 # start token length of Informer decoder
    args.pred_len = 24 # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 13 # encoder input size
    args.dec_in = 13 # decoder input size
    args.c_out = 13 # output size
    args.factor = 5 # probsparse attn factor
    args.d_model = 512 # dimension of model
    args.n_heads = 8 # num of heads
    args.e_layers = 2 # num of encoder layers
    args.d_layers = 1 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu' # activation
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in ecoder
    args.mix = True
    args.padding = 0
    args.freq = 'h'

    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 6
    args.patience = 3
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    # Set augments by using data name
    data_parser = {
        'TELCO':{'data':'TELCO_data.csv','T':'target','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    # Create model
    n_features = X_train.shape[1]
    out_dim = X_train.shape[1]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    Exp = Exp_Informer

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

        # set experiments
        exp = Exp(args)

        # train
        print(setting)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

    # When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
    # The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)
    import numpy as np
    preds = np.load('./results/'+setting+'/pred.npy')
    trues = np.load('./results/'+setting+'/true.npy')

    print('Mae: ', mae(preds, trues))
    print('Mse: ', mse(preds, trues))
    print('Rmse: ', rmse(preds, trues))

    # TODO: Save model and prediction in the proper folder


if __name__ == "__main__":
    params = load_params(PARAMS_FILE, type="yaml")

    main(params)
