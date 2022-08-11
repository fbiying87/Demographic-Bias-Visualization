import pandas as pd
import numpy as np
from AUC_code_example import draw_fdr

def calculate_global_fmnr(pos_dists, neg_dists):
    fnmrs_list = []
    threshold_list = []

    fmr = 1
    neg_dists = np.sort(neg_dists)
    idx = len(neg_dists) - 1
    num_query = len(pos_dists)
    while idx >= 0:
        thresh = neg_dists[idx]
        num_acc = sum(pos_dists < thresh)
        fnmr = 1.0 * (num_query - num_acc) / num_query

        if fmr == 1e-1:
            sym = ' ' if thresh >= 0 else ''
            line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
            print(line)
            fnmrs_list.append(fnmr)
            threshold_list.append(thresh)
        elif fmr == 1e-2:
            sym = ' ' if thresh >= 0 else ''
            line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
            print(line)
            fnmrs_list.append(fnmr)
            threshold_list.append(thresh)
        elif fmr == 1e-3:
            sym = ' ' if thresh >= 0 else ''
            line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
            print(line)
            fnmrs_list.append(fnmr)
            threshold_list.append(thresh)
        elif fmr == 1e-4:
            sym = ' ' if thresh >= 0 else ''
            line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
            print(line)
            fnmrs_list.append(fnmr)
            threshold_list.append(thresh)
        elif fmr == 1e-5:
            sym = ' ' if thresh >= 0 else ''
            line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
            print(line)
            fnmrs_list.append(fnmr)
            threshold_list.append(thresh)

        if idx == 0:
            break
        idx /= 10
        idx = int(idx)
        fmr /= float(10)

    return threshold_list

def merge_files(filenames, database, method):
    # Open file3 in write mode
    with open('./evaluation/{}_{}.txt'.format(database, method), 'w') as outfile:
        # Iterate through list
        for names in filenames:
            # Open each file in read mode
            with open(names) as infile:
                # read the data from file1 and
                # file2 and write it in file3
                outfile.write(infile.read())

            # Add '\n' to enter data of file2
            # from next line
            outfile.write("\n")

def calculate_fmr_fnmr_by_threshold(pos_dists, neg_dists, thresh):

    num_acc = sum(pos_dists < thresh)
    num_query = len(pos_dists)
    fnmr = 1.0 * (num_query - num_acc) / num_query
    fmr = 1.0 * (len(neg_dists) - sum(neg_dists > thresh)) / len(neg_dists)

    line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {:.10f}'.format(fnmr, fmr, thresh)
    print(line)

    return fmr, fnmr

def main():
    tau = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5']

    result_file = open("results.txt", "w")
    result_fdr_auc = open("results_auc.txt", "w")

    for database in ["bfw", "rfw"]:
        tpr_1 = []
        tpr_2 = []
        method_1 = "ArcFace"
        method_2 = "ResNet50"
        for method in [method_1, method_2]:
            df = pd.read_csv('./evaluation/{}_{}.txt'.format(database, method), delimiter=' ',
                             header=None,
                             names=['dist', 'label'])
            pos_dist = df['dist'][df['label'] == 1]
            neg_dist = df['dist'][df['label'] == 0]
            threshold_list = calculate_global_fmnr(np.array(pos_dist), np.array(neg_dist))
            print(threshold_list)

            for index, thresh in enumerate(threshold_list):
                fmr_list = []
                fnmr_list = []
                for ethnicity in ["Caucasian", "Asian", "African", "Indian"]: # ["males", "females"]:
                    print(database, method, ethnicity)
                    df_ethnicity = pd.read_csv('./evaluation/{}_{}_{}.txt'.format(database, method, ethnicity),
                                               delimiter=' ', header=None,
                                               names=['dist', 'label'])
                    pos_dist_ethnicity = df_ethnicity['dist'][df_ethnicity['label'] == 1]
                    neg_dist_ethnicity = df_ethnicity['dist'][df_ethnicity['label'] == 0]
                    fmr, fnmr = calculate_fmr_fnmr_by_threshold(pos_dist_ethnicity, neg_dist_ethnicity, thresh)
                    result_file.write(
                        "{},{},{},{},{:.10f},{:.10f}\n".format(database, method, ethnicity, tau[index], fmr, fnmr))
                    fmr_list.append(fmr)
                    fnmr_list.append(fnmr)
                A = -np.Inf
                B = -np.Inf
                cls_num = 2
                for i in range(cls_num):
                    for j in range(i + 1, cls_num):
                        A_temp = np.abs(fmr_list[i] - fmr_list[j])
                        if A < A_temp:
                            A = A_temp
                        B_temp = np.abs(fnmr_list[i] - fnmr_list[j])
                        if B < B_temp:
                            B = B_temp
                FDR = 1 - (0.5 * A + 0.5 * B)
                print('FDR@ {}, {}'.format(tau[index], FDR))
                result_fdr_auc.write("{},{},{},{}\n".format(database, method, tau[index], FDR))
                if method == method_1:
                    tpr_1.append(FDR)
                else:
                    tpr_2.append(FDR)
        auc_1, auc_2 = draw_fdr(tpr_1, tpr_2, method_1="ArcFace r100", method_2="ArcFace r50", database=database)
        result_fdr_auc.write("{},{},{},{},{}\n".format(database, method_1, method_2, auc_1, auc_2))
    result_file.close()
    result_fdr_auc.close()

if __name__ == '__main__':
    # database = "bfw"
    # method = "ResNet50"
    #
    # filenames = ['./evaluation/{}_{}_Asian.txt'.format(database, method), './evaluation/{}_{}_African.txt'.format(database, method),
    #              './evaluation/{}_{}_Caucasian.txt'.format(database, method), './evaluation/{}_{}_Indian.txt'.format(database, method)]
    # merge_files(filenames, database, method)

    main()
