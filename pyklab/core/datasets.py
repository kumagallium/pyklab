import urllib.error
import urllib.request
from bs4 import BeautifulSoup
import zipfile
import os
import shutil
import pandas as pd
from matminer.datasets import load_dataset


class Datasets:

    datapath = "datasets/"

    def __init__(self, dbname="starrydata", dtype="interpolated", filetype="csv"):
        self.dbname = dbname
        self.dtype = dtype
        self.filetype = filetype

    def info(self):
        if self.dbname == "starrydata":
            print("Database is '"+self.dbname+"("+self.dtype+")'")
        else:
            print("Database is '"+self.dbname+"'")

    def get_versions(self):
        base_url = "https://github.com"
        file_url = base_url + "/starrydata/starrydata_datasets/tree/master/datasets"
        html = urllib.request.urlopen(file_url)
        soup = BeautifulSoup(html, "html.parser")
        datalist = []
        for a in soup.findAll("a", attrs={"class": "Link--primary"}):
            datalist.append(a["href"])
        versionlist = sorted([dl.split("/")[-1].split(".")[0] for dl in datalist if ".zip" in dl], reverse=True)
        return versionlist

    def starrydata_download(self, version="last"):
        base_url = "https://github.com"
        file_url = base_url + "/starrydata/starrydata_datasets/tree/master/datasets"
        html = urllib.request.urlopen(file_url)
        soup = BeautifulSoup(html, "html.parser")
        datalist = []
        for a in soup.findAll("a", attrs={"class": "Link--primary"}):
            datalist.append(a["href"])
        if version == "last":
            zippath = base_url + sorted([dl for dl in datalist if ".zip" in dl], reverse=True)[0]
        else:
            versionidx = self.get_versions().index(version)
            zippath = base_url + sorted([dl for dl in datalist if ".zip" in dl], reverse=True)[versionidx]
        zippath = zippath.replace("/blob/", "/raw/")
        print(self.datapath)

        if not os.path.exists(self.datapath+self.dtype+"_starrydata_"+version+".csv"):
            if not os.path.exists(self.datapath):
                os.mkdir(self.datapath)
            save_path = self.datapath + "download.zip"

            print("download " + zippath)
            try:
                with urllib.request.urlopen(zippath) as download_file:
                    data = download_file.read()
                    with open(save_path, mode='wb') as save_file:
                        save_file.write(data)
            except urllib.error.URLError as e:
                print(e)

            with zipfile.ZipFile(self.datapath + "download.zip") as obj_zip:
                obj_zip.extractall(self.datapath)

            dirname = zippath.split("/")[-1].split(".")[0]
            if self.dtype == "interpolated":
                shutil.copyfile(self.datapath+dirname+"/"+dirname+"_interpolated_data.csv", self.datapath+self.dtype+"_starrydata_"+version+".csv")
            elif self.dtype == "raw":
                shutil.copyfile(self.datapath+dirname+"/"+dirname+"raw_data.csv", self.datapath+self.dtype+"_starrydata_"+version+".csv")
            shutil.rmtree(self.datapath+dirname)
            os.remove(self.datapath + "download.zip")
        print("finished: " + self.datapath+self.dtype+"_starrydata_"+version+".csv")

    def get_alldata(self, version="last"):
        if self.dbname == "starrydata":
            self.starrydata_download(version)
            try:
                if self.filetype == "csv":
                    df_data = pd.read_csv(self.datapath+self.dtype+"_starrydata_"+version+".csv", index_col=0)
            except:
                df_data = pd.DataFrame([])
        elif self.dbname == "materials project":
            if self.filetype == "csv":
                if not os.path.exists(self.datapath+"mp_all_20181018.csv"):
                    if not os.path.exists(self.datapath):
                        os.mkdir(self.datapath)
                    df_data = load_dataset("mp_all_20181018")
                    df_data.to_csv(self.datapath+'mp_all_20181018.csv')
                else:
                    df_data = pd.read_csv(self.datapath+'mp_all_20181018.csv')

        return df_data

