import matplotlib.pyplot as plt
from matplotlib import collections
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error, r2_score

class Visualization:

    image_dirpath = "images/"

    def plot_init(self):
        plt.rcParams['font.size'] = 11
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.width'] = 1.2
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['axes.grid.axis'] = 'both'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.grid']=True
        plt.rcParams["axes.edgecolor"] = 'black'
        plt.rcParams['grid.linestyle']='--'
        plt.rcParams['grid.linewidth'] = 0.3
        plt.rcParams["legend.markerscale"] = 2
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.framealpha"] = 1
        plt.rcParams["legend.edgecolor"] = 'black'

    def pandas2plot(self, df_data, x, y, c=None, kind="scatter", legend=False, tooltip=None, xmin=None, xmax=None, ymin=None, ymax=None, xlabel=None, ylabel=None, colorbar=False, aspect=False, image_name=""):
        xmin = df_data[x].min() if xmin is None else xmin
        xmax = df_data[x].max() if xmax is None else xmax
        ymin = df_data[y].min() if ymin is None else ymin
        ymax = df_data[y].max() if ymax is None else ymax
        xlabel = x if xlabel is None else xlabel
        ylabel = y if ylabel is None else ylabel

        fig = plt.figure(figsize=(3.5, 3), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        #plots = df_data.plot(kind="scatter", x=x, y=y, c=c, cmap="jet", alpha=0.5, lw=0, ax=ax, colorbar=colorbar)
        if kind == "scatter":
            plots = ax.scatter(x=df_data[x], y=df_data[y], c=c, cmap="jet", alpha=0.5, lw=0)
        elif kind == "line":
            line_kinds = df_data[c].unique()
            #cmap = plt.get_cmap("jet")
            lines = []
            for i, kind in tqdm(enumerate(line_kinds)):
                if legend is True:
                    ax.plot(df_data[df_data[c]==kind][x].values, df_data[df_data[c]==kind][y].values, lw=1, alpha=0.5, label=kind)
                else:
                    ax.plot(df_data[df_data[c]==kind][x].values, df_data[df_data[c]==kind][y].values, lw=1, alpha=0.5, c="red")
        if legend is True:
            ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, facecolor='white', framealpha=1).get_frame().set_linewidth(0.5)


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if aspect:
            aspect = (xmax-xmin)/(ymax-ymin)
            ax.set_aspect(aspect, anchor="SW")
        plt.tight_layout()

        if tooltip is not None:
            annot = ax.annotate("", xy=(0, 0), xytext=(5, 5), fontsize=5, textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind):
                i = ind["ind"][0]
                pos = plots.get_offsets()[i]
                annot.xy = pos
                text = df_data[tooltip].values[i]
                annot.set_text(text)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = plots.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        if image_name == "":
            fig.savefig(self.image_dirpath + x.replace(" ", "_") + "_" + y.replace(" ", "_") + ".png")
        else:
            fig.savefig(self.image_dirpath + image_name + ".png")

    
    def show_parityplot(self, df_data, x, y, c=None, tooltip=None, xmin=None, xmax=None, ymin=None, ymax=None, xlabel=None, ylabel=None, alpha=0.7, colorbar=False, aspect=False, image_name=""):

        data_x = df_data[x]
        data_y = df_data[y]

        xmin = data_x.min() if xmin is None else xmin
        xmax = data_x.max() if xmax is None else xmax
        ymin = data_y.min() if ymin is None else ymin
        ymax = data_y.max() if ymax is None else ymax
        xlabel = x if xlabel is None else xlabel
        ylabel = y if ylabel is None else ylabel

        fig = plt.figure(figsize=(3.5, 3), dpi=300, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        t_min = 0
        t_max = xmax

        mape = (np.sum(np.abs(data_y-data_y)/data_y)/len(data_y))
        ax.plot([t_min-mape, t_max+mape*2], [t_min-mape, t_max+mape*2], alpha=0.1, lw=1, c="k")
        plots = ax.scatter(data_x, data_y, s=10, c=c, alpha=alpha, lw=0)

        if aspect:
            aspect = (xmax-xmin)/(ymax-ymin)
            ax.set_aspect(aspect, anchor="SW")
        plt.tight_layout()

        if tooltip is not None:
            annot = ax.annotate("", xy=(0, 0), xytext=(7, 7), fontsize=5, textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind):
                i = ind["ind"][0]
                pos = plots.get_offsets()[i]
                annot.xy = pos
                text = df_data[tooltip].values[i]
                annot.set_text(text)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = plots.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        if image_name == "":
            fig.savefig(self.image_dirpath + x.replace(" ", "_") + "_" + y.replace(" ", "_") + "_parityplot.png")
        else:
            fig.savefig(self.image_dirpath + image_name + ".png")

    def show_tree_importances(self,model, columns):
        display(pd.DataFrame([model.feature_importances_],columns=columns))

        fig = plt.figure(figsize=(4, 2), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        plots = pd.DataFrame([model.feature_importances_],columns=columns).T.sort_values(0, ascending=False).plot(kind="bar",ax=ax, fontsize=5)

        ax.get_legend().remove()
        plt.tight_layout()
        plt.show()

    def show_regression_errors(self, df_data,x,y):
        data_x = df_data[x]
        data_y = df_data[y]
        mae = mean_absolute_error(data_x, data_y)
        mse = mean_squared_error(data_x, data_y)
        rmse = np.sqrt(mean_squared_error(data_x, data_y))
        r2 = r2_score(data_x, data_y)
        mape = np.mean(np.abs((data_y - data_x) / data_x)) * 100
        rmsle = np.sqrt(np.sum((np.log(data_y+1)-np.log(data_x+1))**2)/len(data_x))
        errors = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape, "RMSLE": rmsle}
        display(pd.DataFrame([errors]))




