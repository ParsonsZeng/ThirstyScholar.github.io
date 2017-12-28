import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

plt.style.use('seaborn-paper')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file_name', help='file name to read from', type=str)
args = parser.parse_args()

script_path = os.path.dirname(__file__)

meta = []
for i in range(10):
    filename = os.path.join(script_path, '{0}_{1}.pkl'.format(args.file_name, i))
    print(filename)

    with open(filename, 'rb') as handle:
        temp = pickle.load(handle)
        meta.append(temp)


meta = np.array(meta)
print(meta.shape)

with open(os.path.join(script_path, '{0}_meta.pkl'.format(args.file_name)), 'wb') as handle:
    pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Rank by mean, choose first 50%
# mean = np.mean(meta, axis=1)
# rank = mean.argsort().argsort()
# print(mean)
# print(rank)
#
# index = []
# for i, r in enumerate(rank):
#     if r >=5: index.append(i)
# print(index)
#
# meta = meta[index, :]


mean = np.mean(meta, axis=0)
std = np.std(meta, axis=0)
max = np.max(meta, axis=0)
min = np.min(meta, axis=0)


plt.title('Mean Return +/- 1 Standard Deviation Over 10 Trials')
plt.plot(range(1, len(mean)+1), mean, label='mean')
plt.fill_between(range(1, len(mean)+1), mean-std, mean+std, alpha=.2, label='+/- std')
plt.ylabel('return')
plt.xlabel('episode')
plt.legend(loc=4)
plt.tight_layout()
plt.show()
