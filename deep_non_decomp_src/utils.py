import matplotlib.pyplot as plt


def twin_plot(time, meas1, meas2, name1, name2,
              xlim1=0.5, xlim2=1, ylim1=-0.6, ylim2=-0.3):
    fig, ax1 = plt.subplots()
    ax1.semilogx(time, meas1, 'b+-')
    ax1.set_xlabel('C ')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(name1, color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim([xlim1, xlim2])
    ax2 = ax1.twinx()
    ax2.plot(time, meas2, 'r+-')
    ax2.set_ylabel(name2, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim([ylim1, ylim2])
    fig.tight_layout()
    plt.show()
    return plt
