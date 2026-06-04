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


def filter_bdnb_individual(dep_code, force):
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
    
    # Définition du chemin de sauvegarde du DataFrame en .csv
    output_folder = os.path.join('data', 'BDNB', 'filter_bdnb_individual')
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    
    # Définition du nom du fichier final
    save_name = f'bdnb_filter_individual_dep{dep_code}'
    
    if save_name not in existing_files or force:

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
        bdnb_dpe_logement.dropna(inplace = True) # certains DPE n'ont pas de surface associée (2021) # todo : enlever aussi si pas d'adresse ?
        
        bdnb_join_id_dpe = bdnb_batiment_groupe_compile.merge(bdnb_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id')
        
        bdnb_filter_individual = bdnb_join_id_dpe.merge(bdnb_dpe_logement, how='inner', on='identifiant_dpe') # on conserve seulement les DPE méthode 3CL 2021 correspondant logements indiv
        doublons = bdnb_filter_individual[bdnb_filter_individual.duplicated(subset = ["identifiant_dpe"], keep=False)] # pour observer les id_dpe en doublons (car correspondants à plusieurs bâtiments à la fois)
        # display(doublons)
        bdnb_filter_individual.drop_duplicates(subset = ["identifiant_dpe"], keep=False, inplace = True) # pour supprimer tous les id_dpe associés à plusieurs bâtiments à la fois
        
        
        bdnb_filter_individual.to_csv(os.path.join(output_folder, save_name), index = False)

    
    else:
        bdnb_filter_individual = pd.read_csv(os.path.join(output_folder, save_name), parse_dates=['date_etablissement_dpe'], date_format='%Y-%m-%d %H:%M:%S')

            
    return bdnb_filter_individual



def filter_manipulated(bdnb_df, surface_gap = 1, period = 30): # todo: rajouter condition sur surface ?
# todo : exclure DPE identiques fait le meme jour = doublons ? verifier que meme infos détaillées ou pas
    """
    Identification des bâtiments ayant calculés des DPE .

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        Comparaison entre deux DPEs successifs d'un même bâtiment   
        Colonnes :
            - batiment_groupe_id : identifiant du batiment
            - first_epc_id : identifiant du premier DPE calculé
            - first_epc_surf : surface renseignée lors du calcul du 1er DPE
            - first_epc : classe du premier DPE calculé (compris entre ‘A’ et ‘G’)
            - second_epc_id : identifiant du deuxième DPE calculé
            - second_epc_surf : surface renseignée lors du calcul du 2e DPE
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
                    'adresse_brut': group.iloc[i]['adresse_brut'],
                    'first_epc_id': group.iloc[i]['identifiant_dpe'],
                    'first_epc_date': group.iloc[i]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'first_epc': group.iloc[i]['classe_bilan_dpe'],
                    'first_epc_surf' : group.iloc[i]['surface_habitable_logement'],
                    'second_epc_id': group.iloc[i+1]['identifiant_dpe'],
                    'second_epc_date': group.iloc[i+1]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'second_epc': group.iloc[i+1]['classe_bilan_dpe'],
                    'second_epc_surf' : group.iloc[i+1]['surface_habitable_logement']
                }) #'batiment_groupe_id': group.name, # groupby fait des Series (cf ci-dessous)

        return pd.DataFrame(consecutive_pairs)

    df_epc_evolution = df_sorted.groupby('batiment_groupe_id').apply(extract_consecutive_dpe, period=period)
    
    # df_epc_evolution = df_epc_evolution.reset_index(drop=True) # pour nettoyer colonne inutile
    
    
    # if surface_condition: 
        
    
    return df_epc_evolution   
    


def plot_heatmap(dep_code, frequency):
    """
    Formate un DataFrame de comparaison de DPE successifs en un DataFrame 2D comptant les évolutions entre classes.

    Parameters
    ----------
    df_epc_evolution : pandas DataFrame
        DataFrame de comparaison de DPE successifs (issu de filter_manipulated)
    frequency : boolean
        if True, trace la heatmap en fréquence et non en absolu

    Returns
    -------
    df_heatmap : pandas DataFrame
        Matrice des transitions entre classes de DPE successifs.
        Colonnes = 1er DPE, Lignes = 2e DPE.
    """
    
    departement = Departement(dep_code)

    # Définition du chemin de sauvegarde des heatmap
    output_folder_heatmap = os.path.join('output', 'heatmap')
    os.makedirs(output_folder_heatmap, exist_ok=True)
    existing_files = os.listdir(output_folder_heatmap)
    
    
    bdnb_df = filter_bdnb_individual(dep_code, force=False) # prend du temps je pense
    df_epc_evolution = filter_manipulated(bdnb_df, surface_gap = 1, period = 30)
    
    
    # Décompte de la fréquence des transitions avec crosstab()
    df_heatmap = pd.crosstab(
        index=df_epc_evolution['second_epc'],  # Lignes = 2e DPE
        columns=df_epc_evolution['first_epc'],  # Colonnes = 1er DPE
    )
    
    # Remplir les classes manquantes (A à G) avec 0
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df_heatmap = df_heatmap.reindex(index=classes, columns=classes, fill_value=0)
    
    
    if frequency:
        df_heatmap = (df_heatmap/len(df_epc_evolution))*100
        df_heatmap = df_heatmap.round(1)
        

    # Pour afficher seulement les valeurs non nulles de la matrice
    annot = df_heatmap
    annot = np.round(annot, 1)
    annot = np.where(annot != 0, annot, "") # todo: modifier pour garder que cases > 0.1 ?


    # Tracé de la figure    
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)

    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    
    if frequency:
        ax = sns.heatmap(df_heatmap, ax=ax, vmin=0, vmax=25, annot=annot, fmt="", cmap='bone_r', cbar=True, cbar_ax=cbar_ax, cbar_kws={'label':'Percentage (%)'})
    else:
        ax = sns.heatmap(df_heatmap, ax=ax, annot=annot, fmt="", cmap='bone_r', cbar_ax=cbar_ax,cbar=True,cbar_kws={'label':"Nombre d'observations"})
    
    ax.set_title(f'{departement.name} - {departement.code}, N={len(df_epc_evolution)}')
    # ax.yaxis.set_inverted(True)
    # ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(True)
    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
    ax.set_ylabel('Second EPC')
    ax.set_xlabel('First EPC')
    
    # Enregistrement de la figure
    save_name = f'DPE_manipulation_classes_{dep_code}.png'
    if frequency:
        save_name = save_name.replace('.png','_frequency.png')

    plt.savefig(os.path.join(output_folder_heatmap,save_name), bbox_inches='tight')
    
    
    plt.show()


    return df_heatmap



def plot_pysankey(df_epc_evolution):
    
    sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

    return 



def plotly_sankey(dep_code):
    
    departement = Departement(dep_code)
    
    bdnb_df = filter_bdnb_individual(dep_code, force=False) # prend du temps je pense
    df_epc_evolution = filter_manipulated(bdnb_df, surface_gap = 1, period = 30)
    
        
    # Compter les transitions entre chaque paire de classes DPE
    transition_counts = df_epc_evolution.groupby(['first_epc', 'second_epc']).size().reset_index(name='count')
    
    # Créer les labels pour les nœuds (7 initiaux + 7 finaux)
    labels = [f"{cls}_initial" for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G']] + [f"{cls}_final" for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
    
    # Créer un mapping entre les classes et leurs indices
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    # Préparer les sources, cibles et valeurs pour le Sankey
    sources = transition_counts['first_epc'].map(lambda x: label_to_index[f"{x}_initial"])
    targets = transition_counts['second_epc'].map(lambda x: label_to_index[f"{x}_final"])
    values = transition_counts['count']
    
    # Associer chaque noeud à la couleur de sa classe initiale
    node_colors = etiquette_colors_dict.values()
    # Convertir en format RGBA (ajout de l'opacité à 0.8 par exemple)
    node_colors_rgba = [
        f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
        for r, g, b in node_colors
        ]
    
    # Associer chaque lien à la couleur de sa classe initiale
    link_colors = transition_counts['first_epc'].map(etiquette_colors_dict)
    # Convertir en format RGBA (ajout de l'opacité à 0.8 par exemple)
    link_colors_rgba = [
        f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
        for r, g, b in link_colors
        ]
    
    # Créer le diagramme de Sankey
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[label.replace("_initial", "").replace("_final", "") for label in labels],  # Affiche juste A, B, C, etc.
            color= node_colors_rgba + node_colors_rgba  # Couleurs différentes pour les deux côtés
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors_rgba 
        )
    ))
    
    # Personnaliser la disposition pour séparer les deux groupes de nœuds
    fig.update_layout(
        title_text=f"Transitions entre classes de DPE ({departement.name} - {departement.code}, N={len(df_epc_evolution)})",
        font_size=12,
    )
    
    fig.show()
    
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


    
    # graphe de passage heatmap
    if True:
        dep_code = '85'
        
        plot_heatmap(dep_code, frequency=True)
        
        
    
    # Sankey diagram
    if False:

        bdnb_df = filter_bdnb_individual(dep_code) # prend du temps je pense
        
        df_epc_evolution = filter_manipulated(bdnb_df)
        
        fig = sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

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
        save_name = 'sankey_diagram_evolution_dpe_successifs_{dep_code}'
        fig.savefig(save_name, bbox_inches="tight", dpi=150)

    
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()
