from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_spade(fName, lim=100, lw=2):
    pdfName = fName + '_QMean.pdf'
    p_bench_file = fName + 'p_bench.npz'
    bench_file = fName + 'ANN_Qmean.npz'
    spade_rew = fName + 'SPADE_Qmean.npz'
    spade_TC = fName + 'SPADE_Qmean_TC.npz'
    struct_ann = fName + 'struct_qmean_.npz'
    rew_data = np.load(spade_rew)['arr_0'][0:lim]
    TC_data = np.load(spade_TC)['arr_0'][0:lim]
    p_bench_data = np.load(p_bench_file)['arr_0'][0:lim, 1]
    bench_data = np.load(bench_file)['arr_0'][0:lim]
    struct_data = np.load(struct_ann)['arr_0'][0:lim]
    print len(struct_data), len(rew_data), len(TC_data)
    plt.plot(rew_data, linewidth=lw, color='r', linestyle='--')
    plt.plot(TC_data, linewidth=lw, color='#a24857', linestyle='-.')
    plt.plot(p_bench_data, linewidth=lw, color='k', linestyle=':')
    plt.plot(bench_data, linewidth=lw, color='c', linestyle='-')
    plt.plot(struct_data, linewidth=lw, color='b')
    plt.legend(['DUPLE', 'DUPLE-NS', 'ANN-p', 'ANN-0-1', 'Struct-ANN'],
               loc='best', fontsize='xx-large')
    plt.ylabel('QMean', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title(fName, fontsize=30)
    plt.tight_layout()
    plt.savefig(pdfName)
    plt.show()
    plt.close()


if __name__ == '__main__':
    fName = sys.argv[1]
    lim = int(sys.argv[2])
    plot_spade(fName, lim=lim, lw=3)
