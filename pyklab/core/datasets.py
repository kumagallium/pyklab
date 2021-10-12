class Datasets:

    def __init__(self, dbname="starrydata", dtype="interpolated"):
        self.dbname = dbname
        self.dtype = dtype

    def info(self):
        print(f"Database is '{self.dbname}({self.dtype})'")

    def getAlldata(self):

        print('getAlldata2')