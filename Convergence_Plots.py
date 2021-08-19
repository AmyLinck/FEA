import matplotlib.pyplot as plt

if __name__ == "__main__":
    functions = ["F3", "F5", "F11", "F17", "F20"]
    methods = ["ODG", "Single", "tree", "tree2", "DG"]
    colors = {"DG": 'k', "ODG": 'b', "Single": 'r', "tree": 'm', "tree2": 'g'}
    linestyles = {"DG": 'solid', "ODG": 'dotted', "Single": 'dashed', "tree": 'dashdot', "tree2": (0, (1, 10))}

    SMALL_SIZE = 8

    plt.rc('font', size=16)  # controls default text sizes
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=16)  # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title

    # fig, axs = plt.subplots(5, figsize=(8,30), sharex='all')
    # fig.suptitle('Convergence Plots')

    fig = plt.figure(figsize=(10, 12))

    ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax5 = plt.subplot2grid((3, 4), (2, 1), colspan=2)
    axs = [ax1, ax2, ax3, ax4, ax5]

    for i, func in enumerate(functions):
        ax = axs[i]
        # ax.figure(i, figsize=(10, 10))
        ax.set_title(func)
        for m in methods:
            f = open(f'./MeetRandom/Graph_{func}_{m}.csv') if (m != 'DG' and m != 'ODG') else open(f'./MeetRandom/Graph2_{func}_{m}.csv')
            x = []
            y = []
            first = True
            for line in f:
                line = line.split(',')
                it = int(line[0])
                score = float(line[1])
                if it == 0 and not first:
                    break
                elif it == 0:
                    first = False
                x.append(it)
                y.append(score)
            print(func)
            print(x)
            print(y)
            ax.set_yscale('log')
            lab = m if m != 'Single' else 'CPSO-S'
            ax.plot(x, y, colors[m], label=lab, linestyle=linestyles[m])
        # plt.legend()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=4)
    # plt.tight_layout(rect=(0,0,1,0.98))
    plt.tight_layout()
    plt.show()
