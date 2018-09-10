import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_amp(fName, lim=200, lw=2):
    p_bench_file = fName + 'ps_bench.npz'
    bench_file = fName + 'ANN_fmeas.npz'
    pg_amp = fName + 'AMP_PG.npz'
    damp = fName + 'ANNAMAP_FMeas_new.npz'
    struct_ann = fName + 'struct_fone_.npz'
    p_bench_data = np.load(p_bench_file)['arr_0'][0:lim, 2]
    bench_data = np.load(bench_file)['arr_0'][0:lim]
    pg_data = np.load(pg_amp)['arr_0'][0:lim]
    amp_data = np.load(damp)['arr_0'][0:lim]
    struct_data = np.load(struct_ann)['arr_0'][0:lim]
    print(amp_data)
    plt.plot(amp_data, linewidth=lw, color='r')
    plt.plot(p_bench_data, linewidth=lw, color='k', linestyle='--')
    plt.plot(bench_data, linewidth=lw, color='c', linestyle='-.')
    plt.plot(pg_data, linewidth=lw, color='g', linestyle=':')
    plt.plot(struct_data, linewidth=lw, color='b')
    plt.legend(['DAME', 'ANN-p', 'ANN-0-1', 'ANN-PG', 'Struct-ANN'],
               loc='best',
               fontsize='xx-large')
    plt.ylabel('F-Measure', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title(fName, fontsize=30)
    plt.tight_layout()
    pdfName = fName + '_F1n.pdf'
    plt.savefig(pdfName)
    plt.show()
    plt.close()


if __name__ == '__main__':
    fName = sys.argv[1]
    lim = int(sys.argv[2])
    plot_amp(fName, lim, lw=3)
