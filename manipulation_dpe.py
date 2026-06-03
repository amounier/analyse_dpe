#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:23:12 2026

@author: audrey
"""


import time
import requests
import io
import os
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from IPython.display import display
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import subprocess
import dask_geopandas 
import re
import warnings
from urllib.request import urlopen, Request
from PIL import Image
from datetime import date
from pyogrio.errors import DataSourceError
import seaborn as sns
import plotly.graph_objects as go
from pySankey.sankey import sankey

from utils import etiquette_colors_dict
from administrative import Departement, France
from download import get_bdnb, draw_local_map, neighbourhood_map, get_batiment_groupe_infos


def filter_bdnb_individual(dep_code):
    """
    Filtrage des DPE 2021 3CL associés à des logements individuels (maisons).

    Parameters
    ----------
    dep_code : str
        code du departement.

    Returns
    -------
    bdnb_filter_individual : pandas DataFrame
        identifiants DPE des logements individuels, associés à leur identifiant bâtiment.
        Colonnes :  batiment_groupe_id  |  ffo_bat_nb_log  |  identifiant_dpe  |  adresse_brut  |  type_batiment_dpe  |  date_etablissement_dpe  |  classe_bilan_dpe  |  surface_habitable_logement
    """
    

    bdnb_dpe_logement, bdnb_rel_batiment_groupe_dpe_logement, bdnb_batiment_groupe_compile = get_bdnb(dep_code)

    # Filtrage des bâtiments correspondants à des logements individuels 
    bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile[bdnb_batiment_groupe_compile.ffo_bat_nb_log == 1][['batiment_groupe_id','ffo_bat_nb_log']]
    bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.compute() 

    bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement[['batiment_groupe_id','identifiant_dpe','adresse_brut']]
    bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement.compute()
    #doublons = bdnb_rel_batiment_groupe_dpe_logement[bdnb_rel_batiment_groupe_dpe_logement.duplicated(subset = ["identifiant_dpe"], keep=False)]


    # Filtrage des DPE arrêté 2021 méthode 3CL logement
    bdnb_dpe_logement = bdnb_dpe_logement[(bdnb_dpe_logement.type_dpe =='dpe arrêté 2021 3cl logement') & (bdnb_dpe_logement.type_batiment_dpe == 'maison')][['identifiant_dpe','type_batiment_dpe', 'date_etablissement_dpe','classe_bilan_dpe','surface_habitable_logement']]
    # bdnb_dpe_logement = bdnb_dpe_logement.set_index('identifiant_dpe') # pourquoi ?
    bdnb_dpe_logement = bdnb_dpe_logement.compute()
    bdnb_dpe_logement.dropna(inplace = True) # certains DPE n'ont pas d'adresse ou de surface associée (2021) # todo : garder quand meme si pas d'adresse ?
    
    bdnb_join_id_dpe = bdnb_batiment_groupe_compile.merge(bdnb_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id')
    
    bdnb_filter_individual = bdnb_join_id_dpe.merge(bdnb_dpe_logement, how='inner', on='identifiant_dpe') # on conserve seulement les DPE méthode 3CL 2021 correspondant logements indiv
    doublons = bdnb_filter_individual[bdnb_filter_individual.duplicated(subset = ["identifiant_dpe"], keep=False)] # pour observer les id_dpe en doublons (car correspondants à plusieurs bâtiments à la fois)
    display(doublons)
    bdnb_filter_individual.drop_duplicates(subset = ["identifiant_dpe"], keep=False, inplace = True) # pour supprimer tous les id_dpe associés à plusieurs bâtiments à la fois
    
    return bdnb_filter_individual



def filter_manipulated(bdnb_df, surface_gap = 1, period = 30): # todo: condition sur surface
    """
    Identification des bâtiments ayant calculés des DPE .

    Returns
    -------
        
    
        Colonnes :
            - batiment_groupe_id : identifiant du batiment
            - first_epc_id : identifiant du premier DPE calculé
            - first_epc : classe du premier DPE calculé (compris entre ‘A’ et ‘G’)
            - second_epc_id : identifiant du deuxième DPE calculé
            - second_epc : classe du deuxième DPE calculé (compris entre ‘A’ et ‘G’).
    """
    
    # on conserve uniquement les bâtiments qui ont plus d'un DPE
    bat_many_dpe = bdnb_df['batiment_groupe_id'].value_counts() # décompte du nb de dpe par bâtiment
    bat_many_dpe = bat_many_dpe[bat_many_dpe >= 2].index # index des batiments avec plusieurs DPE
    bdnb_df = bdnb_df[bdnb_df['batiment_groupe_id'].isin(bat_many_dpe)]
    
    # tri chronologique du dataframe
    df_sorted = bdnb_df.sort_values(by=['batiment_groupe_id', 'date_etablissement_dpe'])
    
        
    def extract_consecutive_dpe(group, period):
        consecutive_pairs = []
        for i in range(len(group)-1):
            date_diff = (group.iloc[i+1]['date_etablissement_dpe'] - group.iloc[i]['date_etablissement_dpe']).days
            if date_diff <= period :
                consecutive_pairs.append({
                    'first_epc_id': group.iloc[i]['identifiant_dpe'],
                    'first_epc': group.iloc[i]['classe_bilan_dpe'],
                    'second_epc_id': group.iloc[i+1]['identifiant_dpe'],
                    'second_epc': group.iloc[i+1]['classe_bilan_dpe']
                }) #'batiment_groupe_id': group.name, # groupby fait des Series (cf ci-dessous)

        return pd.DataFrame(consecutive_pairs)

    df_epc_evolution = df_sorted.groupby('batiment_groupe_id').apply(extract_consecutive_dpe, period=period)
    
    # df_epc_evolution = df_epc_evolution.reset_index(drop=True) # pour nettoyer colonne inutile
    
    return df_epc_evolution   
    



def plot_sankey(df_epc_evolution):
    
    sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

    return 



def download_dpe_details(dpe_id, force=False):
    """
    Téléchargement des fichiers de sorties des DPE (au format XML)

    Parameters
    ----------
    dpe_id : str
        DESCRIPTION.
    force : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    if '{}.xlsx'.format(dpe_id) in os.listdir(os.path.join('data','DPE','XML')) or force:
        return

    try:
        dls = f"https://observatoire-dpe-audit.ademe.fr/pub/dpe/{dpe_id}/xml"
        req = Request(dls)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')

        content = urlopen(req)

        # with open(os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id)), 'wb') as output:
        with open(os.path.join('data','DPE','XML','{}.xml'.format(dpe_id)), 'wb') as output:
            output.write(content.read())
    except HTTPError:
        return
    return




    
    
#%% ===========================================================================
# script principal
# =============================================================================



def main():
        
    tic = time.time()
        
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    
    dep_code = '91'
    departement = Departement(dep_code)
    
    
    # test "type_batiment_dpe” = maison MAIS “nb_log”=!1 
    if False:
        test_dpe_logement, test_rel_batiment_groupe_dpe_logement, test_batiment_groupe_compile = get_bdnb(dep_code)
    
        test_batiment_groupe_compile = test_batiment_groupe_compile[['batiment_groupe_id','ffo_bat_nb_log']]
        test_batiment_groupe_compile = test_batiment_groupe_compile.compute() 
    
        test_rel_batiment_groupe_dpe_logement = test_rel_batiment_groupe_dpe_logement[['batiment_groupe_id','identifiant_dpe','adresse_brut']]
        test_rel_batiment_groupe_dpe_logement = test_rel_batiment_groupe_dpe_logement.compute()
        #doublons = test_rel_batiment_groupe_dpe_logement[test_rel_batiment_groupe_dpe_logement.duplicated(subset = ["identifiant_dpe"], keep=False)]
        
        # Filtrage des DPE arrêté 2021 méthode 3CL logement
        test_dpe_logement = test_dpe_logement[(test_dpe_logement.type_dpe =='dpe arrêté 2021 3cl logement') & (test_dpe_logement.type_batiment_dpe == 'maison')][['identifiant_dpe','type_batiment_dpe', 'date_etablissement_dpe','classe_bilan_dpe','surface_habitable_logement']]
        # test_dpe_logement = test_dpe_logement.set_index('identifiant_dpe') # pourquoi ?
        test_dpe_logement = test_dpe_logement.compute()
        
          
        test_join_id_dpe = test_batiment_groupe_compile.merge(test_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id')
        
        test_filter_individual = test_join_id_dpe.merge(test_dpe_logement, how='inner', on='identifiant_dpe') # on conserve seulement les DPE méthode 3CL 2021 correspondant aux maisons
        df_bug_maison = test_filter_individual[test_filter_individual.ffo_bat_nb_log != 1]


    
    # graphe de passage
    if False:
        dep_code = '75'
        departement = Departement(dep_code)
        
        letter_to_number_dict = {chr(ord('@')+n):n for n in range(1,10)} # chr() renvoie un str ASCII et ord() renvoie un code unicode
    
        data = {'from':['F','G','E','G','F','D','G','G','E','F','G','C'],
                'to':['E','F','D','E','D','C','D','D','C','C','C','B'],
                'number':[482,395,284,273,151,68,59,59,23,10,8,4]}
        data = pd.DataFrame().from_dict(data).set_index(['from','to'])
    
        data_format = {'from':list('ABCDEFG')}
        for letter in list('ABCDEFG'):
            data_format[letter] = [0.]*len(list('ABCDEFG'))
        data_format = pd.DataFrame().from_dict(data_format).set_index('from')
    
        # extraction des valeur issues du ditionnaire data
        for f in list('ABCDEFG'):
            for t in list('ABCDEFG'):
                try:
                    data_format.loc[f,t] = data.loc[(f,t)].number.values[0]/data.number.sum()*100
                except KeyError:
                    continue
    
        # pour afficher seulement les valeurs non nulles
        annot = data_format.values.T
        annot = np.round(annot, 1)
        annot = np.where(annot != 0, annot, "")
    
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
    
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    
        ax = sns.heatmap(data_format.T,ax=ax,annot=annot, fmt="",cmap='bone_r',cbar_ax=cbar_ax,cbar=True,cbar_kws={'label':'Percentage (%)'})
        ax.set_title(f'{departement.name} - {departement.code}, N={data.number.sum()}')
        # ax.yaxis.set_inverted(True)
        # ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(True)
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
        ax.set_ylabel('Second EPC')
        ax.set_xlabel('First EPC')
        plt.savefig(os.path.join(os.path.join('output','heatmap'),f'DPE_manipulation_classes_{dep_code}.png'), bbox_inches='tight')
        plt.show()
    
    
    # Sankey diagram
    if True:

        
        bdnb_df = filter_bdnb_individual(dep_code) # prend du temps je pense
        
        df_epc_evolution = filter_manipulated(bdnb_df)
        
        sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

        # Get current figure
        # fig = plt.gcf()
        
# =============================================================================
#         # Set size in inches
#         fig.set_size_inches(6, 6)
#         
#         # Set the color of the background to white
#         fig.set_facecolor("w")
# =============================================================================
        
        # Save the figure
        save = 'sankey_diagram_evolution_dpe_successifs_{dep_code}'
        fig.savefig(save, bbox_inches="tight", dpi=150)

    
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()
