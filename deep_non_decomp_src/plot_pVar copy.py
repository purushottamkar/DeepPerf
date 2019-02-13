import matplotlib.pyplot as plt
import numpy as np
import sys
fig, ax = plt.subplots()

X = (100 * (np.asarray([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000,
                         0.8000, 0.9000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000]) - 1)).astype(np.int32)
fileN = sys.argv[1]
bench_data = np.loadtxt('./kld_label/' + fileN + '_bqs.csv', delimiter=',')
denims_data = np.loadtxt('./kld_label/' + fileN + '_dqs.csv', delimiter=',')
bd = np.zeros((bench_data.shape[0]))
dd = np.zeros((denims_data.shape[0]))
for i in xrange(bd.shape[0]):
    bd[i] = -bench_data[i][1]
    dd[i] = -denims_data[i][1]

N = 16
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
rects1 = ax.bar(ind, bd, width, color='r')
rects2 = ax.bar(ind + width, dd, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('test KLD')
ax.set_title(
    'Variation of KLD with changing class proportion')
ax.set_xlabel('Percentage change in class proportion')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(X)
ax.legend((rects1[0], rects2[0]), ('ANN-Bench', 'DENIMS'))
plt.savefig(fileN+'_pVar.png')
plt.show()
print bd, dd, X
