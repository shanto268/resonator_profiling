import Labber
import numpy as np
import os
import matplotlib.pyplot as plt
from fitTools.Resonator import Resonator

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def partition_dict(original_dict, n, slice_length):
    return [
        {key: values[i*slice_length : (i+1)*slice_length] for key, values in original_dict.items()}
        for i in range(n)
    ]


def get_res_fits_by_partition(fpath, makePlots=False):
    lf = Labber.LogFile(fpath)
    path, fname = os.path.split(fpath)
    figpath = path+f"/figures/{fname[:-4][:-1]}"
    fits = get_resonator_fits(lf, figpath, fname, makePlots)
    freq_channel_name = lf.getStepChannels()[1]["name"]
    res_freq = lf.getData(name = freq_channel_name)
    n = len(res_freq)
    power_channel_name = lf.getStepChannels()[0]["name"]
    vna_power = lf.getData(name = power_channel_name)
    slice_length = len(vna_power[0])
    fits = partition_dict(fits, n, slice_length)
    return fits

def get_resonator_fits(lf, figpath, fname, makePlots=True):
    nEntries = lf.getNumberOfEntries()

    fits = {'f':[],'Q':[],'Qint':[],'Qext':[],'kappa':[],'P1photon':[]}

    for n in range(nEntries):
        (xdata,ydata) = lf.getTraceXY(entry = n)
        res = Resonator('r',xdata,ydata)
        res.autofit(electric_delay=0)
        fits['f'].append(res.f0*1e-9)
        fits['Q'].append(res.Q)
        fits['Qint'].append(res.Qi)
        fits['Qext'].append(res.Qc)
        fits['kappa'].append(res.kappa)
        fits['P1photon'].append(res.P1photon)

        if makePlots:
            if res.fit_found:
                print(20*"=")
                figname = figpath+"/"+fname[:-4][:-1]+f'resonance_{n}-dBm.png'
                res.show(savefile = figname)
                print(res)
                print(20*"="+"\n")
        else:
            pass

    return fits

def get_resonator_fits_from_file(fpath,  makePlots=True):
    lf = Labber.LogFile(fpath)
    path, fname = os.path.split(fpath)
    figpath = path+f"/figures/{fname[:-4]}"
    create_folder_if_not_exists(figpath)
    fname = os.path.basename(fpath)
    return get_resonator_fits(lf, figpath, fname, makePlots)

def combine_dicts(dict1, dict2):
    return {key: dict1.get(key, []) + dict2.get(key, []) for key in set(dict1) | set(dict2)}
