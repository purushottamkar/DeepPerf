from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_spade(fName, lim=100, lw=2):
    pdfName = fName + '_KLD.pdf'
    p_bench_file = fName + 'pk_bench.npz'
    bench_file = fName + 'ANN_kld.npz'
    nemesis_rew = fName + '_kld_rew.npz'
    nemesis_ns = fName + '_kld.npz'
    struct_ann = fName + 'struct_kldnorm_.npz'
    n = 5
    rew_data = np.load(nemesis_rew)['arr_1'][0:lim]
    rew_data = np.convolve(rew_data, np.ones(n) / n)[n - 1:1 - n]
    TC_data = np.load(nemesis_ns)['arr_1'][0:lim]
    TC_data = np.convolve(TC_data, np.ones(n) / n)[n - 1:1 - n]
    p_bench_data = np.load(p_bench_file)['arr_0'][0:lim, 3]
    p_bench_data = np.convolve(p_bench_data, np.ones(n) / n)[n - 1:1 - n]
    print(p_bench_data)
    bench_data = np.load(bench_file)['arr_0'][0:lim]
    bench_data = np.convolve(bench_data, np.ones(n) / n)[n - 1:1 - n]
    # struct_data = -1 * np.load(struct_ann)['arr_0'][0:lim]
    plt.plot(rew_data[5:], linewidth=lw, color='r', linestyle='-.')
    plt.plot(TC_data[5:], linewidth=lw, color='#a24857', linestyle=':')
    plt.plot(bench_data[5:], linewidth=lw, color='c')
    plt.plot(p_bench_data[5:], linewidth=lw, color='k', linestyle='--')
    # plt.plot(struct_data, linewidth=lw, color='b')
    plt.legend(['DENIM', 'DENIM-NS', 'ANN-0-1', 'ANN-p'],
               loc='best', fontsize='xx-large')
    # plt.ylim((-0.7, 0.1))
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel("Neg KLD", fontsize=20)
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
