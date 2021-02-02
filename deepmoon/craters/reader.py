import pandas as pd

def ReadHeadCraterCSV(filename="catalogues/HeadCraters.csv", sortlat=True):
    """Reads Head et al. 2010 >= 20 km diameter crater catalogue.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of Head et al. csv file.  Defaults to the one in
        the current folder.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    craters = pd.read_csv(filename, header=0,
                          names=['Long', 'Lat', 'Diameter (km)'])
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters

def ReadLROCCraterCSV(filename="catalogues/LROCCraters.csv", sortlat=True):
    """Reads LROC 5 - 20 km crater catalogue CSV.

    Parameters
    ----------
    filename : str, optional
        Filepath and name of LROC csv file.  Defaults to the one in the current
        folder.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    craters = pd.read_csv(filename, header=0, usecols=list(range(2, 6)))
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(inplace=True, drop=True)

    return craters


def ReadLROCHeadCombinedCraterCSV(filelroc="catalogues/LROCCraters.csv",
                                  filehead="catalogues/HeadCraters.csv",
                                  sortlat=True):
    """Combines LROC 5 - 20 km crater dataset with Head >= 20 km dataset.

    Parameters
    ----------
    filelroc : str, optional
        LROC crater file location.  Defaults to the one in the current folder.
    filehead : str, optional
        Head et al. crater file location.  Defaults to the one in the current
        folder.
    sortlat : bool, optional
        If `True` (default), order catalogue by latitude.

    Returns
    -------
    craters : pandas.DataFrame
        Craters data frame.
    """
    ctrs_head = ReadHeadCraterCSV(filename=filehead, sortlat=False)
    # Just in case.
    assert ctrs_head.shape == ctrs_head[ctrs_head["Diameter (km)"] > 20].shape
    ctrs_lroc = ReadLROCCraterCSV(filename=filelroc, sortlat=False)
    ctrs_lroc.drop(["tag"], axis=1, inplace=True)
    craters = pd.concat([ctrs_lroc, ctrs_head], axis=0, ignore_index=True,
                        copy=True)
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    return craters
