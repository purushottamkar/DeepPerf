from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_spade(fName, lim=100, lw=2):
    pdfName = fName + '_MinTPRTNR.pdf'
    p_bench_file = fName + 'p_bench.npz'
    bench_file = fName + 'ANN_MinTPRTNR.npz'
    spade_rew = fName + 'SPADE_MinTPRTNR.npz'
    struct_ann = fName + 'struct_mTPRTNR_.npz'
    rew_data = np.load(spade_rew)['arr_0'][0:lim]
    print rew_data.shape
    p_bench_data = np.load(p_bench_file)['arr_0'][0:lim, 0]
    print(p_bench_data.shape)
    bench_data = np.load(bench_file)['arr_0'][0:lim]
    print(bench_data.shape)
    struct_data = np.load(struct_ann)['arr_0'][0:lim]
    plt.plot(rew_data, linewidth=lw, color='r', linestyle=':')
    print(p_bench_data)
    plt.plot(p_bench_data, linewidth=lw, color='k', linestyle='--')
    plt.plot(bench_data, linewidth=lw, color='c', linestyle='-.')
    plt.plot(struct_data, linewidth=lw, color='b')
    plt.legend(['DUPLE', 'ANN-p', 'ANN-0-1', 'Struct-ANN'],
               loc='best', fontsize='xx-large')
    plt.ylabel('Min TPR TNR', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title(fName, fontsize=30)
    plt.savefig(pdfName)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    fName = sys.argv[1]
    lim = int(sys.argv[2])
    plot_spade(fName, lim=lim, lw=3)
