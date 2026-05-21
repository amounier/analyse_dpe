#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:32:43 2026

@author: amounier
"""

import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

etiquette_colors_dict = {'A':(0, 156, 109),'B':(82, 177, 83),'C':(120, 189, 118),'D':(244, 231, 15),'E':(240, 181, 15),'F':(235, 130, 53),'G':(215, 34, 31)}
etiquette_colors_dict = {k: tuple(map(lambda x: x/255, v)) for k,v in etiquette_colors_dict.items()}

etiquette_ep_dict = {'A':[0,70],'B':[70,110],'C':[110,180],'D':[180,250],'E':[250,330],'F':[330,420],'G':[420,np.inf]}

#etiquette_ep_seuils = [70, 110, 180, 250, 330, 420] # old, format liste
etiquette_ep_seuils = {"A/B": 70, "B/C": 110, "C/D": 180, "D/E": 250, "E/F": 330, "F/G": 420}



def get_extent():
    extent = [-5, 9.8, 41.3, 51.3]
    return extent

def blank_national_map():
    fig = plt.figure(figsize=(7,7), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(get_extent())
    
    #ax.add_feature(cfeature.OCEAN, color='lightgrey',zorder=2)
    ax.add_feature(cfeature.LAND, color='w',zorder=1)
    ax.add_feature(cfeature.COASTLINE,zorder=5)
    ax.add_feature(cfeature.BORDERS,zorder=3)
    return fig,ax


#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    # test carte vide
    if True:
        blank_national_map()
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()