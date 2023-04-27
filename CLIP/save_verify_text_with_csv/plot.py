import matplotlib.pyplot as plt
import numpy as np


# models = ['100K_one_poison_1', 'clean_only', 'clean_only_2', 'poison_only_1',
#             'clean_100k', 'clean_only_1', 'poison_100K', 'poison_only_2']
models = ['100K_cat_single', '100K_truck_single']
for model in models:
    data_all = np.load('%s_all.npy' % model)
    data_target = np.load('%s_target.npy' % model)
    data_intended = np.load('%s_intended.npy' % model)

    x = np.arange(data_all.shape[0])

    plt.figure()
    plt.plot(x, data_all.mean(axis=1), label='clean')
    plt.plot(x, data_intended.mean(axis=1), label='intended')
    plt.plot(x, data_target.mean(axis=1), label='target')

    plt.legend()
    plt.savefig('%s.png' % model)