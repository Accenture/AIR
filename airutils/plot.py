"""
Copyright 2020-2021 Accenture

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--plot-data", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--plot-type", default="calibration",
                    help="Type of the plot to create (default: calibration)")
parser.add_argument("--show", action="store_true",
                    help="Display the plot")
parser.add_argument("--every-nth-row", default=1, type=int,
                    help="Plot every nth datapoint (default: 1)")
parser.add_argument("--start-row", default=0, type=int,
                    help="Starting datapoint (default: 0)")
parser.add_argument("--end-row", default=-1, type=int,
                    help="Ending datapoint (default: -1)")
parser.add_argument("--tick-interval", default=0.1, type=float,
                    help="Interval between plot a axis ticks (default: 5e5)")
parser.add_argument("--out-file", default=None,
                    help="Output file name for the plot image (default: plot.pdf)")
parser.add_argument("--smoothing-window", default=None, type=int,
                    help="Smoothen the output curve by averaging over window of this size (default: no smoothing)")

args = parser.parse_args()

# csv_log_file = os.path.join(model_dir, "log.csv")

# strip out some rows if there's plenty of them
# df_thinned = df.iloc[args.start_row:args.end_row:args.every_nth_row, :]

# filter possible extra headers, drop out some values, divide between agents

# df_cleaned = df[pd.to_numeric(df, errors='coerce').notnull()]

SUBCAP_OFFSET = -0.3 

def trim_plot_box(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


if args.plot_type == "calibration":
    df = pd.read_csv(args.plot_data, header=0)

    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    plt.subplots_adjust(wspace=0.2, bottom=0.2, top=0.95) #, hspace=0.45)
    scores = np.array(pd.to_numeric(df["score_thres"]))
    test_precision = np.array(pd.to_numeric(df["test_precision"]))
    test_recall = np.array(pd.to_numeric(df["test_recall"]))
    test_mAP = np.array(pd.to_numeric(df["test_mAP"]))

    test_precision_mix = np.array(pd.to_numeric(df["test_precision_mix"]))
    test_recall_mix = np.array(pd.to_numeric(df["test_recall_mix"]))
    test_mAP_mix = np.array(pd.to_numeric(df["test_mAP_mix"]))

    test_precision_sar = np.array(pd.to_numeric(df["test_precision_sar"]))
    test_recall_sar = np.array(pd.to_numeric(df["test_recall_sar"]))
    test_mAP_sar = np.array(pd.to_numeric(df["test_mAP_sar"]))

    
    # train_precision = np.array(pd.to_numeric(df["train_precision"]))
    # train_recall = np.array(pd.to_numeric(df["train_recall"]))
    # train_mAP = np.array(pd.to_numeric(df["train_mAP"]))

    # train_precision_sar = np.array(pd.to_numeric(df["train_precision_sar"]))
    # train_recall_sar = np.array(pd.to_numeric(df["train_recall_sar"]))
    # train_mAP_sar = np.array(pd.to_numeric(df["train_mAP_sar"]))

    
    # if args.smoothing_window is not None:
    #     N = args.smoothing_window
    #     crop = N // 2 + 1
    #     mean_start = np.zeros(crop)
    #     std_start = np.zeros(crop)
    #     for j in range(crop):
    #         mean_start[j] = mean_return_per_episode[0:j].dot(np.ones(j)/j)
    #         std_start[j] = return_std[0:j].dot(np.ones(j)/j)
    #     mean_return_per_episode = np.convolve(mean_return_per_episode, np.ones((N,))/N, mode='same')[crop:-crop]
    #     return_std = np.convolve(return_std, np.ones((N,))/N, mode='same')[crop:-crop]
    #     frames = frames[:-crop]
    #     mean_return_per_episode = np.concatenate((mean_start, mean_return_per_episode))
    #     return_std = np.concatenate((std_start, return_std))

    axs[0].plot(scores, test_precision, "o-", label="precision")
    axs[0].plot(scores, test_recall, "^-", label="recall")
    # axs[0,0].plot(scores, test_mAP, "s-", label="AP")
    # plt.fill_between(frames, mean_return_per_episode - return_std,
    #                 mean_return_per_episode + return_std, alpha=0.2)
    axs[0].set_xlabel(r"score threshold $t_s$")
    axs[0].set_ylabel("percent", labelpad=5)
    axs[0].set_title(r"$\bf{(a)}$ VOC2012 & NMS", y=SUBCAP_OFFSET+0.05)

    # axs[0,0].set_xticks(np.arange(scores[0], scores[-1] + args.tick_interval, args.tick_interval))
    axs[0].set_xticks(np.arange(0, 0.6, 0.1))
    axs[0].set_xlim([0,0.55])
    axs[0].set_yticks(np.arange(0, 1.1, 0.2))
    trim_plot_box(axs[0])
    axs[0].legend(loc="lower right")
    axs[0].grid(alpha=0.2)

    axs[1].plot(scores, test_precision_mix, "o-", label="precision")
    axs[1].plot(scores, test_recall_mix, "^-", label="recall")
    # axs[0,1].plot(scores, test_mAP_sar, "s-", label="AP")
    # plt.fill_between(frames, mean_return_per_episode - return_std,
    #                 mean_return_per_episode + return_std, alpha=0.2)
    axs[1].set_xlabel(r"score threshold $t_s$")
    # axs[0,1].set_ylabel("percent")
    axs[1].set_title(r"$\bf{(b)}$ SAR-APD & NMS", y=SUBCAP_OFFSET+0.05)

    # axs[0,1].set_xticks(np.arange(scores[0], scores[-1] + args.tick_interval, args.tick_interval))
    axs[1].set_xticks(np.arange(0, 0.6, 0.1))
    axs[1].set_xlim([0,0.55])
    axs[1].set_yticks(np.arange(0, 1.1, 0.2))
    trim_plot_box(axs[1])
    axs[1].legend(loc="lower right")
    axs[1].grid(alpha=0.2)

    axs[2].plot(scores, test_precision_sar, "o-", label="precision")
    axs[2].plot(scores, test_recall_sar, "^-", label="recall")
    axs[2].set_xlabel(r"score threshold $t_s$")
    # axs[2].set_ylabel("percent", labelpad=5)
    axs[2].set_title(r"$\bf{(c)}$ SAR-APD & MOB", y=SUBCAP_OFFSET+0.05)

    # axs[0,0].set_xticks(np.arange(scores[0], scores[-1] + args.tick_interval, args.tick_interval))
    axs[2].set_xticks(np.arange(0, 0.6, 0.1))
    axs[2].set_xlim([0,0.55])
    axs[2].set_yticks(np.arange(0, 1.1, 0.2))
    trim_plot_box(axs[2])
    axs[2].legend(loc="lower right")
    axs[2].grid(alpha=0.2)

    # axs[1,1].plot(scores, train_precision_sar, "o-", label="precision")
    # axs[1,1].plot(scores, train_recall_sar, "^-", label="recall")

    # axs[1,1].set_xlabel("score threshold")
    # # axs[0,1].set_ylabel("percent")
    # axs[1,1].set_title(r"$\bf{(d)}$" + " SAR-APD train eval with MOB", y=SUBCAP_OFFSET)

    # # axs[0,1].set_xticks(np.arange(scores[0], scores[-1] + args.tick_interval, args.tick_interval))
    # axs[1,1].set_xticks(np.arange(0, 0.6, 0.1))
    # axs[1,1].set_xlim([0,0.55])
    # axs[1,1].set_yticks(np.arange(0, 1.1, 0.2))
    # trim_plot_box(axs[1,1])
    # axs[1,1].legend(loc="lower right")
    # axs[1,1].grid(alpha=0.2)

elif args.plot_type == "pr-curve":
    ''' thanks to https://stackoverflow.com/a/39862264 '''
    json_files = args.plot_data.split(",")
    labels = ["a", "b", "c"]
    aps          = [84.8, 86.5, 91.7]
    prcs         = [90.1, 90.5, 94.9]
    rcls         = [86.1, 87.8, 92.9]
    score_thress = [0.25, 0.25, 0.05]
    titles = [" VOC2012 test evaluation with NMS", " SAR-APD test evaluation with NMS", " SAR-APD test evaluation with MOB"]
    fig, axs = plt.subplots(len(json_files), 1, figsize=(8,9))
    plt.subplots_adjust(wspace=0, hspace=0.5, bottom=0.1, top=0.95)
    for i, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            plot_dict = json.load(f)
        recall = np.array(plot_dict["data"][0]["x"]) * 100
        precision = np.array(plot_dict["data"][0]["y"]) * 100

        # take a running maximum over the reversed vector of precision values, reverse the
        # result to match the order of the recall vector
        decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]

        
        # axs[i].plot(recall, precision, '--b')
        axs[i].step(recall, decreasing_max_precision, '-b') #, label="precision-recall curve")
        axs[i].fill_between(recall, decreasing_max_precision, step="pre", alpha=0.1)
        axs[i].scatter(rcls[i], prcs[i], color="red", label=f"point at score threshold = {score_thress[i]}")
        axs[i].annotate(f"({rcls[i]:.1f}, {prcs[i]:.1f})", (rcls[i], prcs[i]), (5, 5), textcoords='offset points')
        axs[i].legend(loc="upper right")
        trim_plot_box(axs[i])
        axs[i].set_xticks(np.arange(0, 110, 10))
        axs[i].set_xlim([0,100])
        axs[i].set_ylim([88,101])
        axs[i].grid(alpha=0.2)
        axs[i].set_xlabel("recall (%)", x=1, horizontalalignment="right")
        axs[i].set_ylabel("precision (%)", labelpad=5)
        axs[i].set_title(r"$\bf{(" + labels[i] + r")}$" + titles[i], y=SUBCAP_OFFSET-0.1)
        axs[i].text(0.5, 0.4, f"AP = {aps[i]}%", color="blue", alpha=0.6, fontsize=18, horizontalalignment="center", verticalalignment="center", transform=axs[i].transAxes)
        # axs[i].grid(alpha=0.2)
        # ax.set_yticks(np.arange(0, 1.1, 0.2))

if args.plot_type == "tiling-perf":

    df = pd.read_csv(args.plot_data, header=0)
    df_voc = df[df["eval_mode"] == "voc2012"]
    df_heridal = df[df["eval_mode"] == "sar-apd"]

    fig, axs = plt.subplots(2, 2, figsize=(8, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0.3, bottom=0.18, top=0.98)

    tiling_dims = np.array(pd.to_numeric(df_voc["image_tiling_dim"]))
    tiling_dims[tiling_dims == 0] = 1.0
    tiling_dims_sar = np.array(pd.to_numeric(df_heridal["image_tiling_dim"]))
    tiling_dims_sar[tiling_dims_sar == 0] = 1.0

    test_precision = np.array(pd.to_numeric(df_voc["test_precision_person"]))
    test_recall = np.array(pd.to_numeric(df_voc["test_recall_person"]))

    test_precision_sar = np.array(pd.to_numeric(df_heridal["test_precision_person"]))
    test_recall_sar = np.array(pd.to_numeric(df_heridal["test_recall_person"]))

    test_time = np.array(pd.to_numeric(df_voc["test_mean_inference_time"]))
    test_time_sar = np.array(pd.to_numeric(df_heridal["test_mean_inference_time"]))
    

    
    # axs[0,0].plot(tiling_dims, test_mAP_sar, "o-", label="precision")
    axs[0,0].plot(tiling_dims, test_time, "^-", label="average execution time")
    # axs[0,0].plot(scores, test_mAP, "s-", label="AP")
    # plt.fill_between(frames, mean_return_per_episode - return_std,
    #                 mean_return_per_episode + return_std, alpha=0.2)
    axs[0,0].set_xlabel("tiling dimension")
    axs[0,0].set_ylabel("average execution time (s)", labelpad=5)
    # axs[0,0].set_title(r"$\bf{(a)}$" + " VOC2012 test eval with NMS", y=SUBCAP_OFFSET)

    axs[0,0].set_xticks(np.arange(0, 11.0, 1.0))
    axs[0,0].set_xlim([0.5, 10.5])
    axs[0,0].set_ylim([0.0, 4.5])
    trim_plot_box(axs[0,0])
    # axs[0,0].legend(loc="lower right")
    axs[0,0].grid(alpha=0.2)

    

    axs[0,1].plot(tiling_dims_sar, test_time_sar, "^-", label="average execution time")
    axs[0,1].set_xlabel("tiling dimension")
    axs[0,1].set_ylabel("average execution time (s)", labelpad=5)
    # axs[0,1].set_ylabel("seconds", labelpad=5)
    # axs[0,1].set_title(r"$\bf{(a)}$" + " SAR-APD test eval with MOB", y=SUBCAP_OFFSET)

    axs[0,1].set_xticks(np.arange(0, 11.0, 1.0))
    axs[0,1].set_xlim([0.5, 10.5])
    axs[0,1].set_ylim([0.0, 4.5])
    trim_plot_box(axs[0,1])
    # axs[0,1].legend(loc="lower right")
    axs[0,1].grid(alpha=0.2)

    axs[1,0].plot(tiling_dims, test_precision, "o-", label="precision")
    axs[1,0].plot(tiling_dims, test_recall, "^-", label="recall")
    axs[1,0].set_xlabel("tiling dimension")
    axs[1,0].set_ylabel("percent", labelpad=5)
    axs[1,0].set_title(r"$\bf{(a)}$" + " VOC2012 test evaluation with NMS", y=SUBCAP_OFFSET-0.15)

    axs[1,0].set_xticks(np.arange(0, 11.0, 1.0))
    axs[1,0].set_xlim([0.5, 10.5])
    axs[1,0].set_ylim([0.0, 1.0])
    trim_plot_box(axs[1,0])
    axs[1,0].legend(loc="lower right")
    axs[1,0].grid(alpha=0.2)

    axs[1,1].plot(tiling_dims_sar, test_precision_sar, "o-", label="precision")
    axs[1,1].plot(tiling_dims_sar, test_recall_sar, "^-", label="recall")

    axs[1,1].set_xlabel("tiling dimension")
    axs[1,1].set_ylabel("percent", labelpad=5)
    axs[1,1].set_title(r"$\bf{(b)}$" + " SAR-APD test evaluation with MOB", y=SUBCAP_OFFSET-0.15)

    axs[1,1].set_xticks(np.arange(0, 11.0, 1.0))
    axs[1,1].set_xlim([0.5, 10.5])
    axs[1,1].set_ylim([0.0, 1.0])
    trim_plot_box(axs[1,1])
    axs[1,1].legend(loc="lower right")
    axs[1,1].grid(alpha=0.2)


if args.plot_type == "loss-plot":

    df = pd.read_csv(args.plot_data, header=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.0, hspace=0.45, bottom=0.15, top=0.95)

    epochs = np.array(pd.to_numeric(df["epoch"]))
    loss = np.array(pd.to_numeric(df["loss"]))
    val_loss = np.array(pd.to_numeric(df["val_loss"]))

    reg_loss = np.array(pd.to_numeric(df["regression_loss"]))
    val_reg_loss = np.array(pd.to_numeric(df["val_regression_loss"]))

    class_loss = np.array(pd.to_numeric(df["classification_loss"]))
    val_class_loss = np.array(pd.to_numeric(df["val_classification_loss"]))

    val_losses = [val_class_loss, val_reg_loss, val_loss]
    losses = [class_loss, reg_loss, loss]

    best_epoch = 17

    subtitles = ["Classification (Focal) loss", "Regression (Smooth L1) loss", "Total loss"]
    labels = ["a", "b", "c"]

    for i in range(3):
        max_y = np.max(val_losses[i])
        axs[i].plot(epochs, losses[i], "-", label="training")
        axs[i].plot(epochs, val_losses[i], "-", label="validation")
        axs[i].vlines(best_epoch, 0, max_y + 0.05, linestyle=":", color="red", label="best epoch") #, alpha=0.8)
        axs[i].set_ylim([0.0, max_y + 0.05])
        # axs[i].set_ylabel("loss")
        # if i == 2:
        axs[i].set_xlabel("epoch", x=1, horizontalalignment="right") # loc="right")
        axs[i].set_title(r"$\bf{(" + labels[i] + r")}$ " + subtitles[i], y=SUBCAP_OFFSET - 0.05)
        trim_plot_box(axs[i])
        axs[i].legend(loc="upper right")
        axs[i].grid(alpha=0.2)


if args.plot_type == "meta-analysis":
    labels = [r'$\bf{(a)}$'+'\n28', r'$\bf{(b)}$'+'\n47', r'$\bf{(c)}$'+'\n203']
    # labels = ("group 1", "group 2", "group 3")
    precs = [94, 91, 53]
    recs = [92, 80, 66]
    ats = [44, 53, 29]

    width = 0.2  # the width of the bars
    
    x1 = np.arange(len(labels), dtype=float)  # the label locations
    x2 = x1.copy()
    # x2[0] += width/2; x2[2] -= width/2
    x = [x1, x2]
    
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 7))
    plt.subplots_adjust(wspace=0, hspace=0.6, bottom=0.1, top=0.95)
    for i in range(2):
        if i == 0:
            rects1 = ax[i].bar(x[i] - width/2, precs, width, label='precision')
            rects2 = ax[i].bar(x[i] + width/2, recs, width, label='recall')
        else:
            rects3 = ax[i].bar(x[i], ats, width, color="red", label='average time')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax[i].set_ylabel('percent')
        # ax[i].set_ylim([0, 100])
        # if i == 1:
        ax[i].set_xlabel('number of images inspected by each group')
        if i == 0:
            ax[i].set_ylabel('percent (%)')
        else:
            ax[i].set_ylabel('average time per image (s)')
        # ax[i].set_title('Scores by group and gender')
        ax[i].set_xlim([-2*width, 2+2*width])
        ax[i].set_xticks(x[i])
        ax[i].set_xticklabels(labels, linespacing = 1.5)
        if i == 0:
            ax[i].legend()
        trim_plot_box(ax[i])
        # ax[i].spines["left"].set_visible(False)
        # ax[i].set_yticks([])
        # ax[i].set_yticklabels([])
        # ax[i].grid(alpha=0.2)
        if i == 0:
            ax[i].bar_label(rects1, padding=3)
            ax[i].bar_label(rects2, padding=3)
        else:
            ax[i].bar_label(rects3, padding=3)

    fig.tight_layout()


if args.out_file is None:
    args.out_file = f"{args.plot_type}_plot.pdf"

figpath = os.path.join(SCRIPT_DIR, args.out_file)
plt.savefig(figpath, bbox_inches="tight")
print("Saved figure to:", figpath)

if args.show:
    plt.show()
