import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from itertools import islice
import seaborn as sns
import pandas as pd


def reversed_lines(file):
    "Generate the lines of file in reverse order."
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part: yield part[::-1]


def reversed_blocks(file, blocksize=4096):
    "Generate blocks of file's contents in reverse order."
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while 0 < here:
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)


def check_last_10_lines(file, key):
    t = []
    for line in islice(reversed_lines(file), 1000):
        t.append(line)
        if line.rstrip('\n') == key:
            break
    return t


def split_string(x):
    x = x.split(sep=' ')
    x = [x.strip() for x in x]
    x = list(filter(None, x))
    x = np.array(x, dtype=np.float32)
    return x

def draw_dense_plot(Mcts_results, prediction):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500)
    g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    m = df.g.map(ord)
    df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    for i in [1,2]:
        g = sns.FacetGrid(df * i, row="g", hue="g", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        # set contout to white
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

        #draw x axsis in color
        #passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "x")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.show()

def plt_3_d_histogram(Mcts_results, predictions):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ['r', 'g', 'b', 'y']
    zticks = [0,1,2,3]
    for c, k in zip(colors, zticks):
        MCTS_result = split_string(Mcts_results[232][k])
        MCTS_dis = MCTS_result/ np.sum(MCTS_result)
        prediction = split_string(predictions[232][k])
        x_axis = list(range(len(MCTS_dis)))
        ax.bar(x_axis, MCTS_dis, zs=k, alpha=0.7, color = 'y')
        #ax.bar(x_axis, prediction, zs=k, alpha=0.5,color ='b')


        #ax.stairs(MCTS_dis)
        #ax.twinx()
        # ax.bar(range(len(MCTS_dis)),MCTS_dis, zs=k, alpha=0.8)
        # #ax.stairs(prediction, zs)
        #
        #
        # # Generate the random data for the y=k 'layer'.
        # xs = np.arange(20)
        # ys = np.random.rand(20)
        #
        # # You can provide either a single color or an array with the same length as
        # # xs and ys. To demonstrate this, we color the first bar of each set cyan.
        # cs = [c] * len(xs)
        # cs[0] = 'c'
        #
        # # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        # ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim([0, 1])
    ax.view_init(elev=120, azim=-90, roll=0)

    # On the y-axis let's only label the discrete values that we have data for.
    #ax.set_yticks(yticks)

    plt.show()


if __name__ == '__main__':
    # with open('/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/output (3).log', "r") as file1:
    #     input_string = check_last_10_lines(file1,
    #                     'key')
    MCTS_result = {
        1 : {
            0: '104.  75.  82.  37.  63.  48.  72.  83.  19. 153.  63.  44.  30.  84. 43.',
            1: '96.  74.  82.  33.  61.  44.  70.  88.  19. 159.  67.  45.  31.  87. 44',
            2 : '109.  86. 127.  12.  16.  10.  29.  96.  91. 169.  58.  47.  26.  76.  48.',
            3 : '103. 105. 135.  16.  20.  12.  21.  28.  26.  94.  57.  75.  86. 146. 76.'
        },
        102 : {
            0: '186. 119. 160.   6.   7.   8.   9.  12.   7.  27.  20.  95.  88. 106.  150.',
            1: '169. 123. 113.   7.   9.   9.  17.  40.  23.  74.  63.  73.  87. 126.   67.',
            2: '142.  98. 130.   7.  12.  19.  37.  57.   6. 100.  42.  88.  75. 119.  68.',
            3: '152. 102. 126.  15.  25.  32.  40.  47.  22.  81.  49.  86.  65.  90.  68.'
        },
        161: '407. 140. 195.   4.   4.   3.   4.   8.   6.  10.   9.  73.  17.  17.  103.',
        162: '125.  94. 103.  24.  31.  33.  46.  65.  21.  97.  66.  80.  64.  87.  64.',
        163: {
            0:'96.  75.  90.  17.  26.  26.  55.  74.   2. 156.  69.  97.  56. 113.  48.',
            1:'110. 102. 132.  18.  21.  19.  32.  33.   6.  45.  40.  88. 115. 187. 52.',
            2:'233. 139. 182.   4.   6.   5.  13.  74.  18.  60.  33.  89.  32.  32.  80.',
            3:'263.  87. 269.   4.   4.   4.   6.   7.   7.  19.  15.  92.  51.  52. 120.'
        },
        164: {
            0 : '149. 103. 146.  10.  20.  28.  51.  62.   1. 127.  45.  79.  37. 105.  37.'
        },
        232: {
            0: '308. 180. 122.   3.   5.   5.   5.   8.   1.  18.  10.  51. 112. 125. 47.',
            1: '148.  92. 119.   1.   6.  13.  57.  72.   0. 157.  61. 117.  36.  75. 46.',
            2: '194. 107. 252.   4.   5.   6.   9.  19.  21.  55.  46.  84.  53.  62.  83.',
            3: '276. 149. 187.   2.   4.   5.  10.  20.   0. 102.  23.  69.  40.  65.  48.',
        }
    }

    prediction = {
        1   : {
            0 : '0.09 0.07 0.07 0.05 0.06 0.04 0.06 0.08 0.08 0.14 0.06 0.04 0.03 0.08 0.04',
            1 : '0.09 0.07 0.08 0.05 0.06 0.04 0.06 0.08 0.08 0.14 0.06 0.04 0.03 0.08 0.04',
            2 : '0.09 0.07 0.08 0.05 0.06 0.04 0.06 0.08 0.08 0.15 0.06 0.04 0.03 0.08 0.04',
            3 : '0.09 0.07 0.08 0.05 0.06 0.04 0.06 0.08 0.08 0.15 0.06 0.04 0.03 0.08 0.04'
        } ,
        102: {
            0: '0.15 0.11 0.13 0.02 0.03 0.03 0.03 0.04 0.03 0.06 0.05 0.08 0.07 0.09  0.08',
            1: '0.12 0.09 0.11 0.03 0.03 0.03 0.04 0.06 0.03 0.08 0.06 0.07 0.08 0.11  0.07',
            2: '0.12 0.08 0.11 0.03 0.03 0.03 0.05 0.06 0.02 0.1  0.05 0.08 0.07 0.11 0.06',
            3: '0.15 0.1  0.12 0.02 0.03 0.03 0.04 0.05 0.02 0.08 0.05 0.08 0.07 0.1  0.07'
        },
        161 : '0.31 0.11 0.15 0.01 0.02 0.01 0.02 0.02 0.02 0.04 0.03 0.08 0.03 0.03  0.11',
        162: '0.13 0.1  0.11 0.02 0.03 0.03 0.05 0.06 0.02 0.09 0.06 0.08 0.06 0.09 0.07',
        163: {
            0 : '0.1  0.08 0.09 0.02 0.03 0.03 0.05 0.07 0.01 0.15 0.07 0.09 0.06 0.11 0.05',
            1 : '0.11 0.1  0.13 0.02 0.02 0.02 0.03 0.03 0.03 0.04 0.04 0.08 0.11 0.19 0.05',
            2 : '0.16 0.1  0.14 0.02 0.02 0.02 0.04 0.05 0.03 0.09 0.06 0.09 0.04 0.06 0.09',
            3 : '0.21 0.11 0.17 0.02 0.02 0.02 0.02 0.03 0.03 0.04 0.03 0.08 0.07 0.07 0.09'
        },
        164: {
            0 : '0.13 0.09 0.13 0.01 0.02 0.03 0.05 0.06 0.01 0.13 0.05 0.08 0.04 0.12 0.04'
        },
        232: {
            0: '0.24 0.15 0.16 0.01 0.02 0.02 0.02 0.02 0.01 0.03 0.02 0.07 0.08 0.09 0.06',
            1: '0.14 0.09 0.12 0.01 0.02 0.02 0.05 0.07 0.   0.16 0.06 0.11 0.04 0.08 0.05',
            2: '0.15 0.1  0.14 0.02 0.02 0.02 0.03 0.04 0.03 0.07 0.05 0.09 0.07 0.08 0.09',
            3: '0.21 0.13 0.15 0.01 0.02 0.02 0.04 0.05 0.   0.09 0.04 0.08 0.04 0.08 0.05',
        }
    }
    plt_3_d_histogram(MCTS_result, prediction)
    # draw_dense_plot(
    #     Mcts_results = list(MCTS_result.values()),
    #     prediction = list(prediction.values())
    # )

    # pattern = r'\[(.*?)\]'
    # for line in input_string:
    #     matches = re.findall(pattern, line)

    # MCTS_result = split_string(MCTS_result)
    # MCTS_dis = MCTS_result / np.sum(MCTS_result)
    # prediction = split_string(prediction)
    # fig, ax1 = plt.subplots()
    # ax1.set_ylim([0, 1])
    # ax1.stairs(MCTS_dis)
    # ax1.twinx()
    # ax1.stairs(prediction)
    # ax1.set_ylim([0, 1])
    # plt.show()
