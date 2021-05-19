import pandas as pd


class Craters:
    def __init__(self, sortlat:bool=True):
        self.sortlat = sortlat
        self.__craters = None

    def __call__(self, filename_lroc, filename_head_et_al):
        head = self.__head(filename_head_et_al)
        assert head.shape == head[head["Diameter (km)"] > 20].shape


        lroc = self.__lroc(filename_lroc)
        lroc.drop(["tag"], axis=1, inplace=True)

        self.__craters =  pd.concat([lroc, head], axis=0, ignore_index=True,
                        copy=True)
        if self.sortlat:
            self.__craters.sort_values(by='Lat', inplace=True)

        self.__craters.reset_index(inplace=True, drop=True)

    @property
    def craters(self):
        return self.__craters

    def __lroc(self, filename):
        csv = pd.read_csv(filename, header=0, usecols=list(range(2,6)))

        if self.sortlat:
            csv.sort_values(by='Lat', inplace=True)
            csv.reset_index(inplace=True, drop=True)

        return csv

    def __head(self, filename):
        csv = pd.read_csv(filename, header=0,
                          names=['Long', 'Lat', 'Diameter (km)'])
        if self.sortlat:
            csv.sort_values(by='Lat', inplace=True)
            csv.reset_index(inplace=True, drop=True)

        return csv
