
import pymatgen.core as mg
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_log_error
from sklearn import mixture
from pycaret import regression
from bokeh.sampledata.periodic_table import elements
import collections
import pickle
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import seaborn as sns
from matplotlib.patches import Rectangle

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
    
    def __init__(self, ad_dirpath="adparams/", image_dirpath="images/", error_dirpath="errors/",model_dirpath="models/", other_dirpath="others/"):
        self.ad_dirpath = ad_dirpath
        self.image_dirpath = image_dirpath
        self.error_dirpath = error_dirpath
        self.model_dirpath = model_dirpath
        self.other_dirpath = other_dirpath

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

            # REFACTOR: need to consider file extensions
            with open(self.ad_dirpath+'nn_train.pickle', 'wb') as f:
                pickle.dump(neigh, f)

            with open(self.ad_dirpath+'th_train.pickle', 'wb') as f:
                pickle.dump(ti, f)
        else:
            # REFACTOR: need to consider file extensions
            with open(self.ad_dirpath+'nn_train.pickle', 'rb') as f:
                neigh = pickle.load(f)
            with open(self.ad_dirpath+'th_train.pickle', 'rb') as f:
                ti = pickle.load(f)

        return neigh, ti

    def count_AD(self, nn, df, thlist):
        dists = nn.kneighbors(df, return_distance=True)[0]
        return (dists <= thlist).sum(axis=1)

    def create_ad_starrydata_models(self, targets, df_train, inputsize):
        for target in tqdm(targets):
            train = pd.concat([df_train.iloc[:, :inputsize], df_train[[target]]], axis=1)
            reg_models = regression.setup(train, target=target, session_id=0, silent=True, verbose=False, transform_target=True)  # ,transformation=True,transform_target=True
            selected_model = regression.create_model('rf', verbose=False)
            final_model = regression.finalize_model(selected_model)
            if not os.path.exists(self.model_dirpath):
                os.mkdir(self.model_dirpath)
            regression.save_model(final_model, model_name = self.model_dirpath + 'model_' + target.replace(" ", "_"))

    def create_final_model(self, target, df_data, inputsize):
        df_train_all = df_data.copy()
        df_train_all = pd.concat([df_data.iloc[:, :inputsize], df_data[[target]]], axis=1)
        reg_models = regression.setup(df_train_all, target=target[0], session_id=0, silent=True, verbose=False, transform_target=True)  # ,transformation=True,transform_target=True
        selected_model = regression.create_model('rf',verbose=False)
        #pred_model = regression.predict_model(selected_model)
        final_model = regression.finalize_model(selected_model)
        if not os.path.exists(self.model_dirpath):
            os.mkdir(self.model_dirpath)
        regression.save_model(final_model,model_name = self.model_dirpath + 'model_final_' + target.replace(" ", "_"))

    def get_matfamily_cluster(self, df_data, inputsize, kind="BGM", clusternum=15, covariance_type="tied", random_state=10):
        df_cluster = df_data.copy()
        clusterinputs  = df_cluster.iloc[:,:inputsize].values
        if kind == "BGM":
            ms = mixture.BayesianGaussianMixture(n_components=clusternum, random_state=random_state, init_params="kmeans", covariance_type=covariance_type)  # diag, full,spherical,tied
            ms.fit(clusterinputs)
            labels = ms.predict(clusterinputs)

        if not os.path.exists(self.model_dirpath):
            os.mkdir(self.model_dirpath)
        pickle.dump(ms, open(self.model_dirpath+kind+"model", 'wb'))

        df_cluster["cluster"] = labels
        clusters = np.sort(df_cluster["cluster"].unique())

        matfamily = []
        clusterelements = elements.copy()
        for c in clusters:
            clustercomp = df_cluster[df_cluster["cluster"] == c]["composition"].unique()
            clustereltmp = [0] * len(clusterelements)
            for comp in clustercomp:
                for el, frac in mg.Composition(comp).fractional_composition.as_dict().items():
                    clustereltmp[mg.Element(el).number-1] += 1  # frac
            clusterelements["count"] = clustereltmp
            matfamily.append("-".join(clusterelements.sort_values("count", ascending=False)[:3]["symbol"].values))

        return clusters, matfamily, df_cluster

    def get_year_cluster_list(self,df_cluster, clusters):
        years = np.sort(df_cluster["year"].unique())
        ylist = []
        for c in range(len(clusters)):
            ylist.append([])
            for y in years:
                ylist[c].append(df_cluster[(df_cluster["year"]<=y)&(df_cluster["cluster"]==c)]["cluster"].count())

        return ylist

    def save_numofdata_years_cluster(self, yclist, matfamily, df_cluster):
        years = np.sort(df_cluster["year"].unique())
        fig = plt.figure(figsize=(3.4, 3), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_yscale("log")
        ax.set_xlabel("Published year")
        ax.set_ylabel("Number of training data")
        ax.set_xlim(2001, 2020)
        # ax.set_ylim(1, 3*10**4)
        cmap = plt.get_cmap("tab20c").colors

        ax.stackplot(years, np.array(yclist)[df_cluster["cluster"].value_counts().index.values], labels=np.array(matfamily)[df_cluster["cluster"].value_counts().index.values], colors=cmap)

        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=6.5, facecolor='white', framealpha=1).get_frame().set_linewidth(0.5)
        plt.xticks([2000 + 5*i for i in range(5)])
        plt.tight_layout()

        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        fig.savefig(self.image_dirpath+"numofdata_years_cluster.png")

    def get_matfamily_matcolor(self, df_cluster, matfamily):
        matcolor = {}
        cmap = plt.get_cmap("tab20c").colors
        for idx, mf in enumerate(np.array(matfamily)[df_cluster["cluster"].value_counts().index.values]):
            matcolor.update({matfamily.index(mf): list(cmap)[idx]})

        if not os.path.exists(self.other_dirpath):
            os.mkdir(self.other_dirpath)

        with open(self.other_dirpath+'matcolor', 'wb') as f:
            pickle.dump(np.array(matfamily)[df_cluster["cluster"].value_counts().index.values], f)

        sort_matfamily = np.array(matfamily)[df_cluster["cluster"].value_counts().index.values]
        with open(self.other_dirpath+'matfamily', 'wb') as f:
            pickle.dump(sort_matfamily, f)

        return matcolor, sort_matfamily

    def get_stack_ad_cluster(self, df_data, ad_reliability, clustermodel, clusters, inputsize, tick=20):
        matfamlylist = []
        for i in range(int(ad_reliability.max()/tick)+1):
            relfil = (ad_reliability >= ((tick*(i))))
            if sum(relfil) > 0:
                matfamlylist.append(collections.Counter(clustermodel.predict(df_data[relfil].iloc[:, :inputsize])))

        stack = []
        for idx, c in enumerate(clusters):
            stack.append([])
            for mf in matfamlylist:
                if c in mf:
                    stack[idx].append(mf[c])
                else:
                    stack[idx].append(0)

        return stack

    def get_errors_targets(self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize, tick=20):
        for idx, tg in enumerate(targets):
            if tg == "ZT":
                tg = "Z"
                test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[[tg]]], axis=1)
                test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[[tg]]], axis=1)
                selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                pred_model["Label"] = pred_model["Label"] * pred_model["Temperature"] * 10**-3
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values * pred_model["Temperature"] * 10**-3
            elif tg == "ZTcalc":
                test_inAD_S = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Seebeck coefficient"]]], axis=1)
                selected_model_S = regression.load_model('models/model_Seebeck_coefficient')
                pred_model_S = regression.predict_model(selected_model_S, data=test_inAD_S)
                predAD_S = pred_model_S["Label"].values

                test_inAD_El = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Electrical conductivity"]]], axis=1)
                selected_model_El = regression.load_model('models/model_Electrical_conductivity')
                pred_model_El = regression.predict_model(selected_model_El, data=test_inAD_El)
                predAD_El = pred_model_El["Label"].values

                test_inAD_k = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Thermal conductivity"]]], axis=1)
                selected_model_k = regression.load_model('models/model_Thermal_conductivity')
                pred_model_k = regression.predict_model(selected_model_k, data=test_inAD_k)
                predAD_k = pred_model_k["Label"].values

                predAD = (((predAD_S*10**-6)**2)*(predAD_El)/predAD_k) * df_test_inAD["Temperature"]
                trueAD = df_test_inAD.loc[:, "ZTcalc"].values
            elif tg == "PFcalc":
                test_inAD_S = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Seebeck coefficient"]]], axis=1)
                selected_model_S = regression.load_model('models/model_Seebeck_coefficient')
                pred_model_S = regression.predict_model(selected_model_S, data=test_inAD_S)
                predAD_S = pred_model_S["Label"].values

                test_inAD_El = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Electrical conductivity"]]], axis=1)
                selected_model_El = regression.load_model('models/model_Electrical_conductivity')
                pred_model_El = regression.predict_model(selected_model_El, data=test_inAD_El)
                predAD_El = pred_model_El["Label"].values

                predAD = ((predAD_S*10**-6)**2)*(predAD_El)*(10**3)
                trueAD = df_test_inAD.loc[:, "PFcalc"].values#((df_test_inAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_inAD.loc[:, "Electrical conductivity"])

            else:
                test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[[tg]]], axis=1)
                test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[[tg]]], axis=1)
                selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                predAD = pred_model["Label"].values
                trueAD =df_test_inAD.loc[:, tg].values

            r2list = []
            rmslelist = []
            mapelist = []
            for i in range(int(ad_reliability.max()/tick)+1):
                relfil = (ad_reliability >= ((tick*(i))))
                if sum(relfil) > 0:
                    r2 = r2_score(trueAD[relfil], predAD[relfil])
                    rmsle = mean_squared_log_error(trueAD[relfil], predAD[relfil])  # np.sqrt(np.sum((np.log(predAD[relfil]+1)-np.log(trueAD[relfil]+1))**2)/len(trueAD[relfil]))
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
            ax = fig.add_subplot(2, 2, idx+1)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            if tg == "ZT":
                tg = "Z"
                test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize],df_test_inAD[[tg]]], axis=1)
                test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize],df_test_outAD[[tg]]], axis=1)
                selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                pred_model["Label"] = pred_model["Label"] * pred_model["Temperature"] * 10**-3
                predAD = pred_model["Label"].values
                trueAD =pred_model.loc[:, tg].values

                pred_model_out = regression.predict_model(selected_model, data=test_outAD)
                pred_model_out["Label"] = pred_model_out["Label"] * pred_model_out["Temperature"] * 10**-3
                predAD_out = pred_model_out["Label"].values
                trueAD_out =df_test_outAD.loc[:, tg].values

                ax.set_xlabel("Experimental $zT$")
                ax.set_ylabel("Predicted $zT$")
                t_min = 5
                t_max = -2
                ax.set_xlim(0, 1.5)
                ax.set_ylim(0, 1.5)
            elif tg == "ZTcalc":
                test_inAD_S = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Seebeck coefficient"]]], axis=1)
                test_outAD_S = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[["Seebeck coefficient"]]], axis=1)
                selected_model_S = regression.load_model('models/model_Seebeck_coefficient')
                pred_model_S = regression.predict_model(selected_model_S, data=test_inAD_S)
                predAD_S = pred_model_S["Label"].values
                pred_model_out_S = regression.predict_model(selected_model_S, data=test_outAD_S)
                predAD_out_S = pred_model_out_S["Label"].values

                test_inAD_El = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Electrical conductivity"]]], axis=1)
                test_outAD_El = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[["Electrical conductivity"]]], axis=1)
                selected_model_El = regression.load_model('models/model_Electrical_conductivity')
                pred_model_El = regression.predict_model(selected_model_El, data=test_inAD_El)
                predAD_El = pred_model_El["Label"].values
                pred_model_out_El = regression.predict_model(selected_model_El, data=test_outAD_El)
                predAD_out_El = pred_model_out_El["Label"].values

                test_inAD_k = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Thermal conductivity"]]], axis=1)
                test_outAD_k = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[["Thermal conductivity"]]], axis=1)
                selected_model_k = regression.load_model('models/model_Thermal_conductivity')
                pred_model_k = regression.predict_model(selected_model_k, data=test_inAD_k)
                predAD_k = pred_model_k["Label"].values
                pred_model_out_k = regression.predict_model(selected_model_k, data=test_outAD_k)
                predAD_out_k = pred_model_out_k["Label"].values

                predAD = (((predAD_S*10**-6)**2)*(predAD_El)/predAD_k) * df_test_inAD["Temperature"]
                trueAD = df_test_inAD.loc[:, "ZTcalc"].values

                predAD_out = (((predAD_out_S*10**-6)**2)*(predAD_out_El)/predAD_out_k) * df_test_outAD["Temperature"]
                trueAD_out =df_test_outAD.loc[:, "ZTcalc"].values
                

                ax.set_xlabel("Experimental $zT_{ \mathrm{calc}}$")
                ax.set_ylabel("Predicted $zT_{ \mathrm{calc}}$")
                t_min = 5
                t_max = -2
                ax.set_xlim(0, 1.5)
                ax.set_ylim(0, 1.5)
            elif tg == "PFcalc":
                test_inAD_S = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Seebeck coefficient"]]], axis=1)
                test_outAD_S = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[["Seebeck coefficient"]]], axis=1)
                selected_model_S = regression.load_model('models/model_Seebeck_coefficient')
                pred_model_S = regression.predict_model(selected_model_S, data=test_inAD_S)
                predAD_S = pred_model_S["Label"].values
                pred_model_out_S = regression.predict_model(selected_model_S, data=test_outAD_S)
                predAD_out_S = pred_model_out_S["Label"].values

                test_inAD_El = pd.concat([df_test_inAD.iloc[:, :inputsize], df_test_inAD[["Electrical conductivity"]]], axis=1)
                test_outAD_El = pd.concat([df_test_outAD.iloc[:, :inputsize], df_test_outAD[["Electrical conductivity"]]], axis=1)
                selected_model_El = regression.load_model('models/model_Electrical_conductivity')
                pred_model_El = regression.predict_model(selected_model_El, data=test_inAD_El)
                predAD_El = pred_model_El["Label"].values
                pred_model_out_El = regression.predict_model(selected_model_El, data=test_outAD_El)
                predAD_out_El = pred_model_out_El["Label"].values

                predAD =((predAD_S*10**-6)**2)*(predAD_El)*(10**3)
                trueAD = df_test_inAD.loc[:, "PFcalc"].values
                #trueAD =((df_test_inAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_inAD.loc[:, "Electrical conductivity"])*(10**3)

                predAD_out = ((predAD_out_S*10**-6)**2)*(predAD_out_El)*(10**3)
                trueAD_out = df_test_outAD.loc[:, "PFcalc"].values
                #trueAD_out = ((df_test_outAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_outAD.loc[:, "Electrical conductivity"])*(10**3)

                ax.set_xlabel("Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                ax.set_ylabel("Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                ax.set_xlim(0, 5)
                ax.set_ylim(0, 5)
            else:
                test_inAD = pd.concat([df_test_inAD.iloc[:, :inputsize],df_test_inAD[[tg]]], axis=1)
                test_outAD = pd.concat([df_test_outAD.iloc[:, :inputsize],df_test_outAD[[tg]]], axis=1)
                selected_model = regression.load_model('models/model_'+tg.replace(" ", "_"))
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                predAD = pred_model["Label"].values
                trueAD =df_test_inAD.loc[:, tg].values

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
                elif tg == "PF":
                    ax.set_xlabel("Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                    ax.set_ylabel("Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                    ax.set_xlim(0, 5)
                    ax.set_ylim(0, 5)
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

    def set_matfamily(self, val, matfamily):
        try:
            return matfamily[int(val)]
        except:
            return np.nan

    def get_properties_tables(self, target, df_decriptor_tmp, nn_train, th_train, cluster_model, matfamily, Tmin=300, Tmax=1300, Ttick=100):
        df_decriptor = df_decriptor_tmp.iloc[:,1:].reset_index(drop=True).copy()
        table = {}
        reltable = {}
        clstable = {}
        if target == "ZTcalc":
            model_S = regression.load_model('models/model_Seebeck_coefficient')
            model_k = regression.load_model('models/model_Thermal_conductivity')
            model_El = regression.load_model('models/model_Electrical_conductivity')
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(nn_train, df_decriptor[filterAD].reset_index(drop=True), th_train)
                    new_prediction_S = regression.predict_model(model_S, data=df_decriptor[filterAD].reset_index(drop=True))
                    new_prediction_El = regression.predict_model(model_El, data=df_decriptor[filterAD].reset_index(drop=True))
                    new_prediction_k = regression.predict_model(model_k, data=df_decriptor[filterAD].reset_index(drop=True))
                    new_prediction_S["ZT"] = (((new_prediction_S["Label"]*10**-6)**2)*(new_prediction_El["Label"])/new_prediction_k["Label"]) * new_prediction_S["Temperature"]
                    clusterlist = cluster_model.predict(df_decriptor[filterAD].reset_index(drop=True))
                    #df_test = pd.merge(new_prediction_S, df_decriptor_tmp[filterAD].reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction_S["composition"] = df_decriptor_tmp[filterAD]["composition"].values

                    idx = 0
                    for comp, value in new_prediction_S[["composition", "ZT"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        elif target == "PFcalc":
            model_S = regression.load_model('models/model_Seebeck_coefficient')
            model_El = regression.load_model('models/model_Electrical_conductivity')
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(nn_train, df_decriptor[filterAD].reset_index(drop=True), th_train)
                    new_prediction_S = regression.predict_model(model_S, data=df_decriptor[filterAD].reset_index(drop=True))
                    new_prediction_El = regression.predict_model(model_El, data=df_decriptor[filterAD].reset_index(drop=True))
                    new_prediction_S["PF"] = ((new_prediction_S["Label"]*10**-6)**2)*(new_prediction_El["Label"]) * 10**3
                    clusterlist = cluster_model.predict(df_decriptor[filterAD].reset_index(drop=True))
                    #df_test = pd.merge(new_prediction_S, df_decriptor_tmp[filterAD].reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction_S["composition"] = df_decriptor_tmp[filterAD]["composition"].values

                    idx = 0
                    for comp, value in new_prediction_S[["composition", "PF"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        else:
            if target == "ZT":
                model = regression.load_model('models/model_Z')
            else:
                model = regression.load_model('models/model_'+target.replace(" ","_"))
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(nn_train, df_decriptor[filterAD], th_train)
                    new_prediction = regression.predict_model(model, data=df_decriptor[filterAD].reset_index(drop=True))
                    if target == "ZT":
                        new_prediction["Label"] = new_prediction["Label"] * (10**-3) * new_prediction["Temperature"]
                    elif target == "Electrical conductivity":
                        new_prediction["Label"] = new_prediction["Label"] * (10**-5)
                    clusterlist = cluster_model.predict(df_decriptor[filterAD].reset_index(drop=True))
                    #df_test = pd.merge(new_prediction, df_decriptor_tmp.reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction["composition"] = df_decriptor_tmp[filterAD]["composition"].values

                    idx = 0
                    for comp, value in new_prediction[["composition", "Label"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        df_clstable_tmp = pd.DataFrame(clstable).T

        df_clstable = df_clstable_tmp.applymap(self.set_matfamily,matfamily=matfamily).copy()

        return pd.DataFrame(table).T, pd.DataFrame(reltable).T, df_clstable

    def get_mape(self, val, mapedlist, tick):
        try:
            return mapedlist[int(val / tick)]
        except:
            return np.nan

    def get_mape_table(self, target, df_reltable, tick):
        mapedlist = []
        with open(self.error_dirpath+'mapelist_'+target.replace(" ", "_")+'.pickle', 'rb') as f:
            mapedlist=pickle.load(f)

        df_mape = df_reltable.applymap(self.get_mape, mapedlist=mapedlist, tick=tick).copy()

        return df_mape

    def show_ranking_table(self, df_table, df_mape, df_reltable, df_clstable, df_leaningdata,filrel=50, rank=20, Tmin=300, Tmax=900, Ttick=100, height=2.5, width=4, imagename="", ascending=False):
        df_table_max = (df_table + (df_table * (df_mape/100))).copy()
        df_table_max = df_table_max.applymap(lambda x: '{:.3g}'.format(x))
        df_table_min = (df_table - (df_table * (df_mape/100))).copy()
        df_table_min[df_table_min < 0] = 0
        df_table = df_table[df_table_min > 0]
        df_table_min = df_table_min.applymap(lambda x: '{:.3g}'.format(x))

        mpcomps = list(df_table.index)
        dftop = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopval = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftoprel = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopcls = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopmax = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopmin = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        for T in range(Tmin, Tmax+Ttick, Ttick):
            dftop[T] = list(df_table.sort_values(by=[T], ascending=ascending).index)
            index = df_table.sort_values(by=[T], ascending=ascending)[T].index
            dftoprel[T] = list(df_reltable.loc[index, T].values)
            dftopcls[T] = list(df_clstable.loc[index, T].values)
            dftopmax[T] = list(df_table_max.loc[index, T].values)
            dftopmin[T] = list(df_table_min.loc[index, T].values)
            dftopval[T] = list(df_table.sort_values(by=[T], ascending=ascending)[T].values)

        dftoprel = dftoprel.fillna(0)
        dftopcls = dftopcls.fillna(0)

        dftop_filrel = dftop[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)
        dftopval_filrel = dftopval[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)
        dftopcls_filrel = dftopcls[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)
        dftoprel_filrel = dftoprel[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)
        dftopmax_filrel = dftopmax[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)
        dftopmin_filrel = dftopmin[dftoprel > filrel].apply(lambda s: pd.Series(s.dropna().tolist()),axis=0)

        dftop_filrel = dftop_filrel.fillna("")
        #dftopval_filrel = dftopval_filrel.fillna(0)
        dftoprel_filrel = dftoprel_filrel.fillna(0)
        dftopcls_filrel = dftopcls_filrel.fillna(0)

        learning_materials = df_leaningdata["composition"].unique()

        dftop.index = dftop.index + 1
        dftopval.index = dftopval.index + 1
        dftoprel.index = dftoprel.index + 1
        dftopcls.index = dftopcls.index + 1
        dftopmin.index = dftopmin.index + 1
        dftopmax.index = dftopmax.index + 1
        dftop = dftop[dftopval > 0]

        dftop_filrel.index = dftop_filrel.index + 1
        dftopval_filrel.index = dftopval_filrel.index + 1
        dftoprel_filrel.index = dftoprel_filrel.index + 1
        dftopcls_filrel.index = dftopcls_filrel.index + 1
        dftopmin_filrel.index = dftopmin_filrel.index + 1
        dftopmax_filrel.index = dftopmax_filrel.index + 1
        dftop_filrel = dftop_filrel[dftopval_filrel > 0]


        plt.rcParams['font.size'] = 4.2
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams["axes.grid"] = False

        fig = plt.figure(figsize=(width, height), dpi=400, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params(pad=1)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(bottom="off", top="off")
        ax.tick_params(left="off")
        ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)

        temprange = []
        for T in range(Tmin, Tmax + Ttick, Ttick):
            temprange.append(str(T)+" K")

        dfstr = "" + dftop_filrel + "\n<" + dftopcls_filrel.astype(str) + ", " + dftoprel_filrel.astype(int).astype(str) + ", " + dftopmin_filrel.round(1).astype(str)+"~"+dftopmax_filrel.round(1).astype(str) + ">"
        sns.heatmap(dftopval_filrel.loc[:rank,Tmin:Tmax], annot=dfstr.loc[:rank,Tmin:Tmax],fmt = '', annot_kws={"size": 2.5}, cmap='jet', cbar_kws={"pad":0.01,"aspect":50}, vmin=min(dftopval_filrel.min().values), vmax=max(dftopval_filrel.max().values),yticklabels=1,xticklabels=temprange)

        for i, T in enumerate(tqdm(range(Tmin, Tmax+Ttick, Ttick))):
            uniqcomp = []
            for mat in learning_materials:
                if len(dftop_filrel.loc[:rank, T][dftop_filrel.loc[:rank, T] == mat].index) > 0:
                    if mat not in uniqcomp:
                        ax.add_patch(Rectangle((i, dftop_filrel.loc[:rank, T][dftop_filrel.loc[:rank, T]==mat].index[0]-1), 1, 1, fill=False, edgecolor='grey', lw=0.5))
                        ax.add_patch(Rectangle((i, dftop_filrel.loc[:rank, T][dftop_filrel.loc[:rank, T]==mat].index[0]-1), 1, 1, fill=True, edgecolor=None, facecolor='grey', alpha=0.8, lw=0))
                        uniqcomp.append(mat)
            uniqcomp = []

        plt.tight_layout()  # pad=0.4, w_pad=1, h_pad=1.0)

        if imagename != "":
            if not os.path.exists(self.image_dirpath):
                os.mkdir(self.image_dirpath)
            fig.savefig(self.image_dirpath+imagename+".png")

        plt.show()

