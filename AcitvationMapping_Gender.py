import os
import glob
import numpy as np
from matplotlib import pyplot
from PIL import Image
from zipfile import ZipFile
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2
import pandas as pd

def get_item(img_path):

    transform = transforms.Compose([
                            transforms.Resize(112),
                            transforms.ToTensor()
                            ])

    torch_img = Image.open(img_path)
    torch_img = transform(torch_img)
    return torch_img

def load_stored_data(data):
    data = torch.from_numpy(data)
    data_mean = torch.mean(data, 0, True)
    data_median, _ = torch.median(data, 0, True)
    data_std_mad = torch.std((data-data_median)**2, dim=0, keepdim=True)
    data_std = torch.std(data, dim=0, keepdim=True)
    return data_mean, data_std, data_median, data_std_mad

def get_heatmap(cam):
    cam = F.interpolate(cam, size=(112, 112), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    # this step changes the value range to [0, 255]
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    return  heatmap


def plot_spatial_shift(data_high, data_low, method=None, y_title=None, legend_1=None, legend_2=None):

    plt.plot(data_high, color='r', Label=legend_1)
    plt.plot(data_low, color='b', Label=legend_2)
    plt.xlabel(y_title, fontsize=18)
    plt.ylabel("$\\sum(AM-V)$", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}_spatial.jpg".format(method), bbox_inches='tight')
    plt.close()

def plot_qos_shift(data_high, data_low, method=None, y_title=None, legend_1=None, legend_2=None):

    sns.kdeplot(data_high, color='r', shade=False, Label=legend_1)
    sns.kdeplot(data_low, color='b', shade=False, Label=legend_2)
    plt.xlabel(y_title, fontsize=18)
    plt.ylabel('PDF', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}_shift.jpg".format(method), bbox_inches='tight')
    plt.close()
    # plt.show()


def plot_2d_heatmap(heatmap, title=None, append_score=False, score=None, score_2=None):
    if append_score:
        size = 30, 112, 3
        m = np.zeros(size, dtype=np.uint8)
        m = cv2.putText(m, '{:2,.2f} | {:2,.2f}'.format(score, score_2), (1, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        heatmap = np.clip(heatmap.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        heatmap = np.concatenate((heatmap, m), axis=0)
    else:
        heatmap = np.clip(heatmap.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)

    cv2.imwrite("{}.jpg".format(title), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    # cv2.imshow("{}".format(title), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)

def main(image_vectors, method, ethnicity_1, ethnicity_2, database):
    '''
    statistics derived from high quality samples
    '''
    print(method, ethnicity_2)

    output_dir = "./results/{}/gender/".format(method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load("./data/{}/gender/{}_{}_{}.npy".format(method, method, database, ethnicity_1))

    data_high_mean, data_high_std, data_high_median, data_high_std_median = load_stored_data(data)
    print("High: ", torch.max(data_high_std), torch.min(data_high_std))

    '''
       statistics derived from low quality samples
    '''
    data = np.load("./data/{}/gender/{}_{}_{}.npy".format(method, method, database, ethnicity_2))
    data_low_mean, data_low_std, data_low_median, data_low_std_median = load_stored_data(data)
    print("Low: ", torch.max(data_low_std), torch.min(data_low_std))

    if 1:
        # Mean average mapping for high
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_mean.div(data_high_mean.max()), 0)), title="{}/{}_mean".format(output_dir, ethnicity_1))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_std.div(data_high_std.max()), 0)), title="{}/{}_std".format(output_dir, ethnicity_1))

        # Median average mapping for high
        # plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_median.div(data_high_median.max()), 0)), title="{}/{}_median".format(output_dir, ethnicity_1))
        # plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_std_median.div(data_high_std_median.max()), 0)),
        #                 title="{}/{}_std_median".format(output_dir, ethnicity_1))

        # Mean average mapping for low
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_mean.div(data_low_mean.max()), 0)), title="{}/{}_mean".format(output_dir, ethnicity_2))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_std.div(data_low_std.max()), 0)), title="{}/{}_std".format(output_dir, ethnicity_2))

        # Median average mapping for low
        # plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_median.div(data_low_median.max()), 0)), title="{}/{}_median".format(output_dir, ethnicity_2))
        # plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_std_median.div(data_low_std_median.max()), 0)),
        #                 title="{}/{}_std_median".format(output_dir, ethnicity_2))

        '''
        derivation from higher order statistics
        '''
        plot_qos_shift(torch.flatten(data_high_std), torch.flatten(data_low_std), method="{}/{}_{}_diff_std".format(output_dir, ethnicity_1, ethnicity_2), \
                       y_title="std_mean", legend_1=ethnicity_1, legend_2=ethnicity_2)
        plot_qos_shift(torch.flatten(data_high_mean), torch.flatten(data_low_mean), method="{}/{}_{}_diff_mean".format(output_dir, ethnicity_1, ethnicity_2),\
                       y_title="mean", legend_1=ethnicity_1, legend_2=ethnicity_2)

        '''
        sum over x or y directions
        '''
        plot_spatial_shift(torch.flatten(torch.sum(data_high_std, 1)), torch.flatten(torch.sum(data_low_std, 1)),
                       method="{}/{}_{}_diff_x_pos".format(output_dir, ethnicity_1, ethnicity_2), \
                       y_title="x coordinates", legend_1=ethnicity_1, legend_2=ethnicity_2)
        plot_spatial_shift(torch.flatten(torch.sum(data_high_std, 2)), torch.flatten(torch.sum(data_low_std, 2)),
                       method="{}/{}_{}_diff_y_pos".format(output_dir, ethnicity_1, ethnicity_2), \
                       y_title="y coordinates", legend_1=ethnicity_1, legend_2=ethnicity_2)


    '''
    Deviation maps
    '''
    # variance with mean difference
    diff = torch.abs(data_high_std - data_low_std)
    # print("Diff: ", torch.max(diff), torch.min(diff))
    if 0:
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(diff.div(diff.max()), 0)), title="{}/{}_{}_diff_std".format(output_dir, ethnicity_1, ethnicity_2))

        # deviation with median difference
        diff_median = torch.abs(data_high_std_median - data_low_std_median)
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(diff_median.div(diff_median.max()), 0)), title="{}/{}_{}_diff_std_median".format(output_dir, ethnicity_1, ethnicity_2))


    '''overlap with groundtruth image using example images'''
    if 1:
        img_org = get_item(
            img_path=image_vectors["{}_{}".format(database, ethnicity_2)])

        diff_avg = 0.5 * torch.flip(diff, [2]) + 0.5 * diff

        hmp = 0.5 * get_heatmap(torch.unsqueeze(diff_avg.div(torch.max(diff_avg)), 0)) + 1.0 * img_org
        plot_2d_heatmap(hmp, title="{}/{}_{}_diff_std_mean_sample".format(output_dir, ethnicity_1, ethnicity_2))

        plot_2d_heatmap(get_heatmap(torch.unsqueeze(diff_avg.div(torch.max(diff_avg)), 0)), title="{}/{}_{}_diff_std_median".format(output_dir, ethnicity_1, ethnicity_2))

        img_org_ref = get_item(
            img_path=image_vectors["{}_{}".format(database, ethnicity_1)])
        hmp = 0.5 * get_heatmap(torch.unsqueeze(data_high_mean.div(torch.max(data_high_mean)), 0)) + 1.0 * img_org_ref
        plot_2d_heatmap(hmp, title="{}/{}_mean_sample".format(output_dir, ethnicity_1))
        hmp = 0.5 * get_heatmap(torch.unsqueeze(data_low_mean.div(torch.max(data_low_mean)), 0)) + 1.0 * img_org
        plot_2d_heatmap(hmp, title="{}/{}_mean_sample".format(output_dir, ethnicity_2))

if __name__ == '__main__':

    image_vectors = {
                        "BFW_males" : r"E:\BFW_aligned\white_males\n001063\0004_01.jpg",
                        "BFW_females" : r"E:\BFW_aligned\black_females\n000771\0230_01.jpg"}

    ethnicity_1 = "males"
    ethnicity_2 = "females"
    method = "r100"
    database = "BFW"
    main(image_vectors, method, ethnicity_1, ethnicity_2, database)
