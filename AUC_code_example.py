import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def draw_fdr(tpr_1, tpr_2, method_1, method_2, database):
    # fpr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    fpr = np.linspace(1, 0, num=5)
    x = [1, 2, 3, 4, 5]

    auc_1 = metrics.auc(fpr, tpr_1)
    auc_2 = metrics.auc(fpr, tpr_2)

    labels = ['$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$']

    size = 18
    params = {'legend.fontsize': 'large',
              'axes.labelsize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75}
    plt.rcParams.update(params)

    plt.plot(x, tpr_1, label="{}".format(method_1), marker='o', linewidth=3)
    plt.plot(x, tpr_2, label="{}".format(method_2), marker='o', linewidth=3)
    plt.xticks(x, labels)
    # plt.ylim([0.93, 0.96])
    # plt.ylim([0.85, 1])
    plt.ylim([0.95, 1])
    plt.xlabel("$\\tau@FMR_{10^{-x}}$")
    plt.ylabel("$FDR(\\tau)$")
    plt.legend(loc="lower left")
    plt.grid()
    plt.tight_layout()
    plt.savefig("{}_fdr.jpg".format(database), bbox_inches='tight')
    plt.close()
    # plt.show()
    return auc_1, auc_2

if __name__ == "__main__":
    tpr_1 = [0.9912856584222178, 0.9969353121729341, 0.9978917388171167, 0.9900456467443085, 0.9787053416096699]
    # tpr_2 = [0.9906238063777613, 0.9965159560802488, 0.9945418354185248, 0.9870279390874866, 0.9766008670872937]
    fpr = np.linspace(1, 0, num=5)
    auc_1 = metrics.auc(fpr, tpr_1)
    print(auc_1)
    # draw_fdr(tpr_1, tpr_2, method_1="ArcFace", method_2="ResNet", database="rfw")