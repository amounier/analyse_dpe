#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:35:25 2026

@author: amounier
"""

import time
import pandas as pd
import geopandas as gpd
import os
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import date
from shapely.ops import unary_union
import numpy as np

from utils import blank_national_map

adm = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','communes-departement-region.csv'))
adm = adm.dropna(subset=['code_departement'])
adm['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in adm.code_departement]

geo = gpd.read_file(os.path.join('data','INSEE','decoupage_administratif','departements.geojson'))
prf = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','prefectures.csv'))

zcl = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','zones_climatiques.csv'))
zcl['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in zcl.code_departement]

dict_code_dep_name_dep = {c:n for c,n in zip(adm.code_departement,adm.nom_departement)}
dict_name_dep_code_dep = {n:c for c,n in dict_code_dep_name_dep.items()}
dict_code_dep_name_reg = {d:r for d,r in zip(adm.code_departement,adm.nom_region)}
dict_code_dep_geom_dep = {d:g for d,g in zip(geo.code,geo.geometry)}
dict_code_dep_code_zcl = {d:c for d,c in zip(zcl.code_departement,zcl.zone_climatique)}
dict_code_zcl_code_zcw = {e:e[:2] for e in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H2d', 'H3']}
dict_code_zcl_code_zcs = {e:'d' if e[-1]=='3' else e[-1] for e in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H2d', 'H3']}

dict_name_dep_name_prf = {n:p for n,p in zip(prf.Département,prf.Préfecture)}

list_dep_code = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                 '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', 
                 '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', 
                 '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', 
                 '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', 
                 '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', 
                 '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', 
                 '88', '89', '90', '91', '92', '93', '94', '95']

class Departement:
    def __init__(self,dep_code):
        if type(dep_code) == int:
            self.code = "{:02d}".format(dep_code)
        elif type(dep_code) == str and len(dep_code) == 1:
            self.code = "{:02d}".format(int(dep_code))
        elif type(dep_code) == str and len(dep_code) == 2:
            if dep_code in ['2A','2B']:
                self.code = dep_code
            else:
                self.code = "{:02d}".format(int(dep_code))
        else:
            raise(NotImplementedError('Please use the departement code instead.'))
            
        self.name = dict_code_dep_name_dep.get(self.code)
        self.codint = int(self.code.replace('A','01').replace('B','02'))
        self.region = dict_code_dep_name_reg.get(self.code)
        self.geometry = dict_code_dep_geom_dep.get(self.code)
        self.climat = dict_code_dep_code_zcl.get(self.code)
        self.prefecture = dict_name_dep_name_prf.get(self.name)
        
    def __str__(self):
        return '{} ({})'.format(self.name, self.code)
    

class France:
    def __init__(self):
        self.departements = [Departement(e) for e in list_dep_code]
        self.climats = sorted(list(set([e.climat for e in self.departements])))
        self.geometry = unary_union([d.geometry for d in self.departements])
        
        
def draw_departement_map(dict_dep,figs_folder,cbar_min=0,cbar_max=1.,
                         automatic_cbar_values=False, cbar_label=None, 
                         map_title=None,save=None,cmap=None,figax=None,
                         hide_cbar=False,alpha=None,hatches=None,
                         cbar_format=None,cbar_norm=None,cbar_ticks=None,
                         cbar_extend_format=None):
    
    if figax is not None:
        fig,ax = figax
    else:
        fig,ax = blank_national_map()
    
    if cmap is None:
        cmap = matplotlib.colormaps.get_cmap('viridis')
    else:
        cmap = matplotlib.colormaps.get_cmap(cmap)
    
    plotter = pd.DataFrame().from_dict({'departements':dict_dep.keys(),'vals':dict_dep.values()})
    plotter['geometry'] = [d.geometry for d in plotter.departements]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.vals.quantile(0.99)
        cbar_min = plotter.vals.quantile(0.01)
        cbar_extend = 'both'
    else:
        cbar_extend = 'neither'
    
    if cbar_extend_format is not None:
        cbar_extend = cbar_extend_format
    
    if cbar_norm is None:
        norm = matplotlib.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
    elif cbar_norm == 'log': 
        norm = matplotlib.colors.LogNorm(vmin=cbar_min, vmax=cbar_max)
        
    plotter['color'] = norm(plotter.vals)
    plotter['color'] = plotter['color'].apply(cmap)
    
    if hatches is not None:
        plotter_hatches = plotter[plotter.departements.isin(hatches)]
        plotter_nohatches = plotter[~plotter.departements.isin(hatches)]
        
        plotter_hatches.plot(color=plotter_hatches.color, ax=ax, transform=ccrs.PlateCarree(),alpha=alpha,hatch='//',ec='w',lw=0.2)
        plotter_nohatches.plot(color=plotter_nohatches.color, ax=ax, transform=ccrs.PlateCarree(),alpha=alpha)
        plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=0.5)
    else:
        plotter.plot(color=plotter.color, ax=ax, transform=ccrs.PlateCarree(),alpha=alpha)
        plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=0.5)
    
    
    if not all(plotter.color==(0.0, 0.0, 0.0, 0.0)) and not hide_cbar:
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        if cbar_norm is None:
            norm = matplotlib.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
        elif cbar_norm == 'log': 
            norm = matplotlib.colors.LogNorm(vmin=cbar_min, vmax=cbar_max)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        cbar_label_var = cbar_label
        cbar = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend=cbar_extend, extendfrac=0.02,format=None)
        
        if cbar_ticks is not None:
            minor_all = cbar.get_ticks(minor=True)
            
            major_to_plot = [e for e in cbar_ticks if e not in minor_all]
            minor_to_plot = [e for e in cbar_ticks if e in minor_all]
            
            cbar.set_ticks(ticks=major_to_plot,labels=major_to_plot, minor=False)
            cbar.set_ticks(ticks=minor_to_plot,labels=minor_to_plot, minor=True, fontsize='small')
            

    ax.set_title(map_title)
    if save is not None:
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save)),bbox_inches='tight')
    return fig,ax


    
#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    # test de la carte des départements
    if True:
        today = pd.Timestamp(date.today()).strftime('%Y%m%d')
        output_folder = os.path.join('output',today)
        os.makedirs(output_folder, exist_ok=True)
        
        dict_dep = {d:np.random.random() for d in France().departements} 
        draw_departement_map(dict_dep,output_folder,save='test')
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':  
    main()
