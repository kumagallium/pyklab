
import pymatgen.core as mg
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score,mean_absolute_error
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
    error_dirpath = "errors/"

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

    def count_AD(self, nn, df, thlist):
        dists = nn.kneighbors(df, return_distance=True)[0]
        return (dists <= thlist).sum(axis=1)

    def get_errors_targets(self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize, tick=20):
        for idx, tg in enumerate(targets):
            test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[[tg]]], axis=1)
            test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[[tg]]], axis=1)
            selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
            if tg == "Z":
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                pred_model["Label"] = pred_model["Label"] * pred_model["Temperature"] * 10**-3
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values
            else:
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values

            r2list = []
            rmslelist = []
            mapelist = []
            for i in range(int(ad_reliability.max()/tick)+1):
                relfil = (ad_reliability >= ((tick*(i))))
                if sum(relfil) > 0:
                    r2 = r2_score(trueAD[relfil], predAD[relfil])
                    rmsle = np.sqrt(np.sum((np.log(predAD[relfil]+1)-np.log(trueAD[relfil]+1))**2)/len(trueAD[relfil]))
                    mape = (np.sum(np.abs(predAD[relfil]-trueAD[relfil])/trueAD[relfil])/len(trueAD[relfil]))*100
                    r2list.append(r2)
                    rmslelist.append(rmsle)
                    mapelist.append(mape)

            if not os.path.exists(self.error_dirpath):
                os.mkdir(self.error_dirpath)

            if tg == "Z":
                with open(self.error_dirpath+'mapelist_'+tg+'T.pickle', 'wb') as f:
                    pickle.dump(mapelist, f)
                with open(self.error_dirpath+'rmslelist_'+tg+'T.pickle', 'wb') as f:
                    pickle.dump(rmslelist, f)
                with open(self.error_dirpath+'r2list_'+tg+'T.pickle', 'wb') as f:
                    pickle.dump(r2list, f)
            else:
                with open(self.error_dirpath+'mapelist_'+tg.replace(" ", "_")+'.pickle', 'wb') as f:
                    pickle.dump(mapelist, f)
                with open(self.error_dirpath+'rmslelist_'+tg.replace(" ", "_")+'.pickle', 'wb') as f:
                    pickle.dump(rmslelist, f)
                with open(self.error_dirpath+'r2list_'+tg.replace(" ", "_")+'.pickle', 'wb') as f:
                    pickle.dump(r2list, f)

    def save_parityplot_ad_starrydata_targets(self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize):
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

