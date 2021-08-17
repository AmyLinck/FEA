import matplotlib.pyplot as plt

if __name__ == "__main__":
    funtions = ["F3", "F5", "F11", "F17", "F20"]
    methods = ["DG", "ODG", "Single", "tree", "tree2"]
    colors = {"DG": 'k', "ODG": 'b', "Single": 'r', "tree": 'm', "tree2": 'g'}

    for i, func in enumerate(funtions):
        plt.figure(i, figsize=(10, 10))
        plt.title(func)
        for m in methods:
            f = open(f'./MeetRandom/Graph_{func}_{m}.csv')
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
            print(x)
            print(y)
            plt.yscale('log')
            plt.plot(x, y, colors[m], label=m)
        plt.legend()
        plt.show()
