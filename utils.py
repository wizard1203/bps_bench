import numpy as np
import matplotlib.pyplot as plt


def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            label,
            ha='center', va='bottom', rotation=rotation)


class Data_Get_Config(object):
    def __init__(self, dir_path, model, tensor_size, KB, DMLC_PS, batch_size,
        num_iters, nworkers, nservers, worker_id, local_rank, x_data, legend):
        self.dir_path = dir_path
        self.model = model
        self.tensor_size = tensor_size
        if KB:
            self.KB=KB
        else:
            self.KB=0
        self.DMLC_PS = DMLC_PS 
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.nworkers = nworkers
        self.nservers = nservers
        self.worker_id = worker_id
        self.local_rank = local_rank

        self.x_data = x_data

        self.legend = legend
        self.y_data = None

        self.color = None
        self.marker = None
