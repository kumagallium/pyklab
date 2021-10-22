
import pymatgen.core as mg
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pycaret import regression
import pickle
import os
import matplotlib.pyplot as plt

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

class AD:

    ad_dirpath = "adparams/"
    image_dirpath = "images/"

    def get_threshold(self, df, k=5):
        if not os.path.exists(self.ad_dirpath+"nn_train.pickle") and not os.path.exists(self.ad_dirpath+"th_train.pickle"):
            if not os.path.exists(self.ad_dirpath):
                os.mkdir(self.ad_dirpath)

            neigh = NearestNeighbors(n_neighbors=len(df), radius=None)
            neigh.fit(df)
            dij = pd.DataFrame(neigh.kneighbors(df, return_distance=True)[0]).T[1:]
            di_ave = dij[:k].mean()
            q1 = di_ave.quantile(0.25)
            q3 = di_ave.quantile(0.75)
            refval = q3 + 1.5*(q3-q1)
            Ki = dij[dij <= refval].count().values
            ti = (dij[dij <= refval].sum()/Ki)
            mint = ti[ti > 0].min()
            ti = ti.fillna(mint).values

            with open(self.ad_dirpath+'nn_train.pickle', 'wb') as f:
                pickle.dump(neigh, f)

            with open(self.ad_dirpath+'th_train.pickle', 'wb') as f:
                pickle.dump(ti, f)
        else:
            with open(self.ad_dirpath+'nn_train.pickle', 'rb') as f:
                neigh = pickle.load(f)
            with open(self.ad_dirpath+'th_train.pickle', 'rb') as f:
                ti = pickle.load(f)

        return neigh, ti

    def cntAD(self, nn, df, thlist):
        dists = nn.kneighbors(df, return_distance=True)[0]
        return (dists <= thlist).sum(axis=1)

    def save_parityplot_ad_starrydata(self, target, ad_reliability, pred_inAD, true_inAD, pred_outAD, true_outAD):
        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        fig = plt.figure(figsize=(3, 3), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        if target == "Z":
            ax.set_xlabel("Experimental $zT$")
            ax.set_ylabel("Predicted $zT$")
            t_min = 5
            t_max = -2
            ax.set_xlim(0,1.5)
            ax.set_ylim(0,1.5)
        else:

            if target == "Thermal conductivity":
                ax.set_xlabel("Experimental $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                ax.set_ylabel("Predicted  $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                ax.set_xlim(0, 7)
                ax.set_ylim(0, 7)
            elif target == "Seebeck coefficient":
                ax.set_xlabel("Experimental $S$ [\u03BCVK$^{-1}$]")
                ax.set_ylabel("Predicted $S$ [\u03BCVK$^{-1}$]")
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 500)
            elif target == "Electrical conductivity":
                true_inAD = true_inAD/1000000
                pred_inAD = pred_inAD/1000000
                true_outAD = true_outAD/1000000
                pred_outAD = pred_outAD/1000000
                ax.set_xlabel("Experimental $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]")
                ax.set_ylabel("Predicted  $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]")
                ax.set_xlim(0, 0.6)
                ax.set_ylim(0, 0.6)
            t_min = 0#trueAD.min()
            t_max = true_inAD.max()
        mape = (np.sum(np.abs(pred_outAD - true_outAD) / true_outAD) / len(true_outAD))
        ax.plot([t_min-mape, t_max+mape*2], [t_min-mape, t_max+mape*2], alpha=0.1, lw=1, c = "k")

        true_inAD_viz = np.array([true_inAD.max()+100])
        true_inAD_viz = np.append(true_inAD_viz, true_inAD)
        pred_inAD_viz = np.array([pred_inAD.max()+100])
        pred_inAD_viz = np.append(pred_inAD_viz, pred_inAD)
        ad_reliability_viz = np.array([1])
        ad_reliability_viz_tmp = ad_reliability.copy()
        ad_reliability_viz_tmp = (ad_reliability_viz_tmp-ad_reliability_viz_tmp.min())/(ad_reliability_viz_tmp.max()-ad_reliability_viz_tmp.min())
        ad_reliability_viz = np.append(ad_reliability_viz, ad_reliability_viz_tmp)
        np.place(ad_reliability_viz, ad_reliability_viz == 0, ad_reliability_viz.min())
        ax.scatter(true_inAD_viz, pred_inAD_viz, s=10, c="r", alpha=ad_reliability_viz, lw=0, label="Inside AD")
        ax.scatter(true_outAD, pred_outAD, s=10, c="b", alpha=0.8, lw=0, marker="^", label="Outside AD")

        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, facecolor='white', framealpha=1).get_frame().set_linewidth(0.5)

        plt.tight_layout()
        fig.savefig(self.image_dirpath+target.replace(" ", "_")+".png")

    def save_parityplot_ad_starrydata_all(self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize):
        fig = plt.figure(figsize=(6, 6), dpi=300, facecolor='w', edgecolor='k')
        for idx, tg in enumerate(targets):
            test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize],df_test_inAD[[tg]]], axis=1)
            test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize],df_test_outAD[[tg]]], axis=1)
            ax = fig.add_subplot(2, 2, idx+1)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
            if tg == "Z":
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                pred_model["Label"] = pred_model["Label"] * pred_model["Temperature"] * 10**-3
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values

                pred_model_out = regression.predict_model(selected_model, data=test_outAD)
                pred_model_out["Label"] = pred_model_out["Label"] * pred_model_out["Temperature"] * 10**-3
                predAD_out = pred_model_out["Label"].values
                trueAD_out =pred_model_out.loc[:, tg].values

                ax.set_xlabel("Experimental $zT$")
                ax.set_ylabel("Predicted $zT$")
                t_min = 5
                t_max = -2
                ax.set_xlim(0, 1.5)
                ax.set_ylim(0, 1.5)
            else:
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values

                pred_model_out = regression.predict_model(selected_model, data=test_outAD)
                predAD_out = pred_model_out["Label"].values
                trueAD_out =pred_model_out.loc[:, tg].values

                if tg == "Thermal conductivity":
                    ax.set_xlabel("Experimental $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                    ax.set_ylabel("Predicted  $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                    ax.set_xlim(0, 7)
                    ax.set_ylim(0, 7)
                elif tg == "Seebeck coefficient":
                    ax.set_xlabel("Experimental $S$ [\u03BCVK$^{-1}$]")
                    ax.set_ylabel("Predicted $S$ [\u03BCVK$^{-1}$]")
                    ax.set_xlim(0, 450)
                    ax.set_ylim(0, 450)
                elif tg == "Electrical conductivity":
                    trueAD = trueAD/1000000
                    predAD = predAD/1000000
                    trueAD_out = trueAD_out/1000000
                    predAD_out = predAD_out/1000000
                    ax.set_xlabel("Experimental $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]")
                    ax.set_ylabel("Predicted  $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]")
                    ax.set_xlim(0, 0.6)
                    ax.set_ylim(0, 0.6)
                t_min = 0
                t_max = trueAD.max()

            mape = (np.sum(np.abs(predAD-trueAD)/trueAD)/len(trueAD))
            ax.plot([t_min-mape, t_max+mape*2], [t_min-mape, t_max+mape*2], alpha=0.1, lw=1, c="k")

            trueAD_viz = np.array([trueAD.max()+100])
            trueAD_viz = np.append(trueAD_viz, trueAD)
            predAD_viz = np.array([predAD.max()+100])
            predAD_viz = np.append(predAD_viz, predAD)
            ad_reliability_viz = np.array([1])
            ad_reliability_viz_tmp = ad_reliability.copy()
            ad_reliability_viz_tmp = (ad_reliability_viz_tmp-ad_reliability_viz_tmp.min())/(ad_reliability_viz_tmp.max()-ad_reliability_viz_tmp.min())
            ad_reliability_viz = np.append(ad_reliability_viz, ad_reliability_viz_tmp)
            np.place(ad_reliability_viz, ad_reliability_viz == 0, ad_reliability_viz.min())
            ax.scatter(trueAD_viz, predAD_viz, s=10, c="r", alpha=ad_reliability_viz, lw=0, label="Inside AD")
            ax.scatter(trueAD_out, predAD_out, s=10, c="b", alpha=0.8, lw=0, marker="^", label="Outside AD")

            ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, facecolor='white', framealpha=1).get_frame().set_linewidth(0.5)

        plt.tight_layout()
        fig.savefig(self.image_dirpath+"TE_parityplot.png")

