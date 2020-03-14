import numpy as np
import torch
from os.path import exists
from urllib.request import urlopen


def download_data():
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


def get_data(dataset, datadir):
    if dataset=='eeg':
        download_data()
        data = np.loadtxt(datadir + '/eeg.dat', delimiter=',', skiprows=19)
        print("[raw data shape] {}".format(data.shape))
        #data = data[::2, :]
        #data = data[:4800, :]
        data = data[::20, :]
        data = torch.tensor(data[:, :-1]).float()
        print("[data shape after thinning] {}".format(data.shape))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset == 'currency':
        # https://datahub.io/core/exchange-rates
        data = torch.tensor(np.load(datadir + '/currency.npy'))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[currency data shape] {}".format(data.shape))
    elif dataset in ['krakeu', 'krakusd', 'short']:
        # http://www.cryptodatadownload.com/
        if dataset=='short':
            clip, dataset = True, 'krakeu'
        else:
            clip = False
        data = torch.tensor(np.load(datadir + '/' + dataset + '.npy'))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        if clip:
            data = data[0:100, :]
        print("[{} data shape] {}".format(dataset, data.shape))
    elif dataset == 'ercot':
        # https://blog.valohai.com/smart-grids-use-machine-learning-to-forecast-load
        data = np.loadtxt(datadir + '/ercot.csv', skiprows=1, delimiter=',', usecols=(1,2,3,4,5,10))
        data = torch.tensor(data[0:2000, :])
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset == 'house':
        # https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption#
        data = np.loadtxt(datadir + '/house.txt',skiprows=1,delimiter=';',usecols=(2,3,4,5),max_rows=2000)
        data = torch.tensor(data)
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset == 'aq':
        # https://archive.ics.uci.edu/ml/datasets/Air+quality
        data = np.loadtxt(datadir + '/aq.csv', delimiter=';', skiprows=1,
                          usecols=(3,6,7,8,9,10,11,12,13,14))
                          #usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14))
        data = torch.tensor(data[0:749, :])
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[aq data shape] {}".format(data.shape))
        return data
    elif dataset in ['ushum', 'ustemp']:
        # https://www.kaggle.com/selfishgene/historical-hourly-weather-data/download
        data = torch.tensor(np.load(datadir + '/%s.npy' % dataset))
        data = data[-2000:, :]
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[{} data shape] {}".format(dataset, data.shape))
    elif dataset == 'colo':
        # http://coagmet.colostate.edu/station_map.php
        data = torch.tensor(np.load(datadir + '/colotemp.npy'))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[colorado temperature data shape] {}".format(data.shape))
    elif dataset in ['open', 'volume']:
        # https://www.kaggle.com/jessevent/all-crypto-currencies
        data = torch.tensor(np.load(datadir + '/crypto_%s.npy' % dataset))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[crypto {} data shape] {}".format(dataset, data.shape))
    elif dataset=='metals':
        # https://www.amark.com/graphs/silver-spot-prices
        data = torch.tensor(np.load(datadir + '/metals.npy'))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset=='solar':
        # https://datacatalog.worldbank.org/dataset/nepal-solar-radiation-measurement-data
        data = torch.tensor(np.load(datadir + 'nepal.solar.dni.npz')['arr_0'])#[-100000:]
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset=='dow':
        # https://fred.stlouisfed.org/categories/32255/downloaddata
        data = torch.tensor(np.load(datadir + '/dow.npy'))
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
    elif dataset == 'sml':
        # https://archive.ics.uci.edu/ml/datasets/SML2010
        data = np.loadtxt(datadir + '/sml.txt', delimiter=' ', skiprows=1,
                          usecols = (2,3,5,6,7,8,9,10,12,13,14,15,16,17,21,22))
        #data = torch.tensor(data[0:3600, :])
        data = torch.tensor(data[0:2000, :])
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std
        print("[sml data shape] {}".format(data.shape))
    elif dataset in ['oak', 'sf']:
        data = torch.tensor(np.load(datadir + '/%scounts.npy' % dataset))
        data = data.reshape(data.size(0) // 24, 24, -1).sum(-2)
        data = data[0:2919, 0:8]
        data = data.reshape(data.size(0) // 7, 7, -1).sum(-2)
        print("[{} count data shape] {}".format(dataset, data.shape))
        for station in range(8):
            assert data.shape[0] == np.count_nonzero(data[:, station])
        data = data.log()
        data_mean, data_std = data.mean(0), data.std(0)
        data = data - data_mean
        #data = (data - data_mean) / data_std
    elif dataset=='synth':
        D = 10
        omegas = 0.05 * (1.0 + torch.rand(D))
        ts = torch.arange(3600).unsqueeze(-1)
        data = torch.cos(omegas * ts) + 0.03 * torch.randn(ts.size(0), D)
        data_mean, data_std = data.mean(0), data.std(0)
        data = (data - data_mean) / data_std


    T = data.size(0)
    T_train = int(0.8 * T)
    #T_train = int(0.8 * T) if dataset != 'eeg' else 4800
    #T_train = int(0.8 * T) if dataset != 'eeg' else 600
    T_val = (T - T_train) // 2
    T_test = T - T_train - T_val
    return data, T_train, T_val, T_test
