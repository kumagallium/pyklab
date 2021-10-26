import matplotlib.pyplot as plt
import os

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


class Visualization:

    image_dirpath = "images/"

    def pandas2plot(self, df_data, x, y, c=None, xmin=None, xmax=None, ymin=None, ymax=None, colorbar=False, aspect=False, image_name=""):
        xmin = df_data[x].min() if xmin is None else xmin
        xmax = df_data[x].max() if xmax is None else xmax
        ymin = df_data[y].min() if ymin is None else ymin
        ymax = df_data[y].max() if ymax is None else ymax
        fig = plt.figure(figsize=(3.7, 3), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        df_data.plot(kind="scatter", x=x, y=y, c=c, cmap="jet", alpha=0.5, lw=0, ax=ax, colorbar=colorbar)

        if aspect:
            aspect = (xmax-xmin)/(ymax-ymin)
            ax.set_aspect(aspect, anchor="SW")
        plt.tight_layout()
        plt.show()

        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        if image_name == "":
            fig.savefig(self.image_dirpath + x.replace(" ", "_") + "_" + y.replace(" ", "_") + ".png")
        else:
            fig.savefig(self.image_dirpath + image_name + ".png")