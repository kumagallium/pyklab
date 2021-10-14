import urllib.error
import urllib.request
from bs4 import BeautifulSoup
import zipfile
import os
import shutil
import pandas as pd

class Datasets:

    datapath = "datasets/"

    def __init__(self, dbname="starrydata", dtype="interpolated"):
        self.dbname = dbname
        self.dtype = dtype

    def info(self):
        print("Database is '"+self.dbname+"("+self.dtype+")'")

    def setSigFig(val):
        try:
            return '{:.3g}'.format(float(val))
        except:
            return val

    def starrydata_download(self):
        base_url = "https://github.com"
        file_url = base_url + "/starrydata/starrydata_datasets/tree/master/datasets"
        html = urllib.request.urlopen(file_url)
        soup = BeautifulSoup(html, "html.parser")
        datalist = []
        for a in soup.findAll("a",attrs={"class":"Link--primary"}):
            datalist.append(a["href"])
        zippath = base_url + sorted([dl for dl in datalist if ".zip" in dl],reverse=True)[0]
        zippath = zippath.replace("/blob/","/raw/")
        print(self.datapath)

        if os.path.exists(self.datapath+self.dtype+"_starrydata.csv") == False:
            if os.path.exists(self.datapath) == False:
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
                shutil.copyfile(self.datapath+dirname+"/"+dirname+"_interpolated_data.csv", self.datapath+self.dtype+"_starrydata.csv")
            elif self.dtype == "raw":
                shutil.copyfile(self.datapath+dirname+"/"+dirname+"raw_data.csv", self.datapath+self.dtype+"_starrydata.csv")
            shutil.rmtree(self.datapath+dirname)
            os.remove(self.datapath + "download.zip")
        print("finished: " + self.datapath+self.dtype+"_starrydata.csv")


    def getAlldata(self):
        if self.dbname == "starrydata":
            self.starrydata_download()
            try:
                df_data = pd.read_csv(self.datapath+self.dtype+"_starrydata.csv", index_col=0)
            except:
                df_data = pd.DataFrame([])

        return df_data