#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:05:21 2026

@author: audrey
"""

# todo : enlever trucs inutiles
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, zscore # attention, beta (variable aléatoire) =/= sc.beta (fonction)
from scipy.optimize import curve_fit
import scipy.special as sc 
from datetime import date
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

from administrative import  list_dep_code, Departement, France, draw_departement_map
from utils import etiquette_colors_dict,etiquette_ep_dict,etiquette_ep_seuils
from distribution import calcul_bunching_france




def tension_immob_dep():
    """
    Détermination des proportions de logements dans chacune des 5 zones du zonage ABC pour chaque département.
    Rappel : 
    Les 5 zones sont désignées par ordre de déséquilibre décroissant :
        Zone A : grandes agglomérations où la demande est élevée (Lyon, Lille, Marseille, Montpellier, etc.).
        Zone A bis : Paris et sa proche banlieue avec une demande locative extrêmement forte.
        Zone B1 : villes moyennes dynamiques et zones frontalières ou littorales.
        Zone B2 : petites villes ou communes où la tension est modérée.
        Zone C : zones détendues où l’offre dépasse la demande.

    Parameters
    ----------
    path_tension : str
        chemin vers le fichier Excel des tensions locatives par communes.
    path_logements : str
        chemin vers le fichier Excel du nombre de logements par communes.

    Returns
    -------
    df_tension_immob_dep : pandas DataFrame
        Part des logements du département dans chacune des 5 zones 
        Colonnes :  dep_code (index)  |  total_logements  |  part_A = part de logements du département en zone A  |  part_Abis  |  part_B1  |  part_B2  |  part_C  
    """
    # chemin vers le fichier Excel des tensions locatives par communes
    path_tension = os.path.join('data','INSEE','Liste ensemble des communes - Zonage ABC 5 septembre 2025.xlsx')
    
    # chemin vers le fichier Excel du nombre de logements par communes
    path_logements =  os.path.join('data','INSEE','logement-2022.xlsx')
    
    # Import du zonage de la tension locative par commune
    df_zonage_ABC = pd.read_excel(path_tension, usecols=['CODGEO', 'DEP',  'Zonage en vigueur depuis le 5 septembre 2025'])
    df_zonage_ABC = df_zonage_ABC[~df_zonage_ABC['DEP'].isin(['971', '972', '973', '974', '975', '976'])] 
     
    # Import du nombre de logements par communes
    df_logements = pd.read_excel(path_logements, names=['CODGEO', 'LIBGEO', 'LOGEMENTS_2022'], na_values = 'N/A - résultat non disponible', skiprows=3)
    
    # Jointure des deux dataframes
    df_communes = pd.merge(df_zonage_ABC, df_logements, on='CODGEO') #, 'LIBGEO'])
    df_communes.set_index('CODGEO', inplace=True)
    df_communes = df_communes[~df_communes['DEP'].isin(['971', '972', '973', '974', '975', '976'])] # on étudie seulement la France hexagonale


# =============================================================================
#     OLD : BDD Logements avec Nan dans le Cantal 15

#     # Import du nombre de logements par communes
#     df_logements = pd.read_excel(path_logements, usecols='A:E', dtype={'CODGEO':int,'P22_LOG':float} ,skiprows=5) # usecols=['CODGEO', 'DEP', 'LIBGEO', 'P22_LOG'], skiprows=4)
#     df_logements.set_index('CODGEO')
#     df_logements = df_logements[~df_logements['DEP'].isin(['971', '972', '973', '974', '975', '976'])]
#     df_logements = df_logements['P22_LOG'].map(round) # on arrondi le nombre de logements par départements (pourquoi est-ce des nb décimaux ??)
#     
# =============================================================================
    
    # Création du df des tensions immobilières par département
    df_tension_immob_dep = pd.DataFrame(index=list_dep_code) 
    df_tension_immob_dep['departement'] = [Departement(f'{n}') for n in df_tension_immob_dep.index]
    
    # Colonne nombre total de logements par département
    df_tension_immob_dep['total_logements'] = df_communes.groupby('DEP')['LOGEMENTS_2022'].sum()
    
    # Colonnes nombre de logements dans chaque zone par département
    logements_par_zone = df_communes.groupby(['DEP', 'Zonage en vigueur depuis le 5 septembre 2025'])['LOGEMENTS_2022'].sum()
    logements_par_zone = logements_par_zone.unstack(fill_value=0)  # unstack transforme les zones ABC en colonnes, et fill_value remplace les valeurs manquantes par 0
    # todo : l'ajouter aussi au dataframe final ?
    
    # Colonnes parts des logements dans chaque zone par département
    parts_par_zone = logements_par_zone.div(df_tension_immob_dep['total_logements'], axis=0)
    parts_par_zone.columns = [f'part_{col}' for col in parts_par_zone.columns] # on renomme les colonnes
        
    df_tension_immob_dep = df_tension_immob_dep.join(parts_par_zone)
    
    return df_tension_immob_dep
    
    

def get_nb_diagnostiqueur_dep():
    data = pd.read_csv(os.path.join('data','MTE','annuaire-diagnostiqueurs-immobiliers.csv'),sep=';')
    data = data[data['Type de certificat'].str.contains('DPE')]
    data['CP'] = [f"{e:05d}" for e in data.CP]
    data = data[~data.CP.str.startswith('97')] # uniquement territoire hexagonal
    data = data[~data.CP.str.startswith('20')] # hors corse (2A et 2B aggrégé)
    data['dep_code'] = [Departement(e[:2]).code for e in data.CP]
    
    data_count = data.groupby('dep_code')[['Organisme']].count()
    return data_count
    


    
#%% ===========================================================================
# script principal
# =============================================================================



def main():
    
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    if True:
       
        pd.options.display.max_columns = None
        df_tension_immob_dep = tension_immob_dep()
        # print(df_tension_immob_dep)
        
# =============================================================================
#         # to do : Visualisation des parts par département
#         zone = 'A'
#         france = France()
#         dict_dep_zone = {d:0. for d in france.departements} 
#         for dep in france.departements :
#             dep_code = dep.code
#             print(dep)
#             dict_dep_zone[dep] = df_tension_immob_dep[f'part_{zone}']
# =============================================================================
        

    if True :
        
        # Fixation des paramètres de mesure du bunching
        old_built_filter = True
        method='diff_moyenne'
        # method='AMP'
        itv_bunching=10
        window_size=50
            
        
        france_bunching = calcul_bunching_france(output_folder, method, itv_bunching, window_size, old_built_filter, max_xlim = 600, verbose=False)
        
        
        seuils = ['D/E', 'E/F', 'F/G']
        
        # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
        seuils_sans_slash = '_'.join(seuils)
        seuils_sans_slash = seuils_sans_slash.replace("/","")
        
        # implémentation d'une liste du nom exact des colonnes de france_bunching à sélectionner
        noms_colonnes_seuils = []
        for seuil in seuils:
            colonne_seuil = [colonne for colonne in france_bunching.columns if colonne.startswith(f'{seuil}')]
            nom_colonne = colonne_seuil[0]
            noms_colonnes_seuils.append(nom_colonne)

        france_bunching_cut = france_bunching.filter(items = noms_colonnes_seuils)  
        
        france_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'] = france_bunching_cut.sum(axis=1) # on somme sur les lignes des colonnes conservées    
        
        bunching = france_bunching[[f'Somme_seuils_{seuils_sans_slash}_method_{method}']]
        
        # Vecteur des paramètres
        variables = df_tension_immob_dep
        variables = sm.add_constant(variables)
        
        # variables_list = ['part_A','part_Abis','part_B1','part_B2','part_C']
        variables_list = ['part_A','part_C','zcl_Tref','log_total_logements','Organisme']
        # variables_list = ['zcl_H3','log_total_logements']
        
        bunching = bunching.join(variables)
        bunching = bunching.join(get_nb_diagnostiqueur_dep())
        
        # ajout de la zone climatique 
        bunching["zcl"] = [Departement(e).climat for e in bunching.index]
        bunching["zcl_Tref"] = [Departement(e).climat_Tref for e in bunching.index]
        bunching['zcl_H3'] = (bunching.zcl == 'H3').map(int)
        
        # ajout de l'inverse du nombre de logements
        bunching['log_total_logements'] = np.log(bunching.total_logements)
        
        bunching = bunching[[f'Somme_seuils_{seuils_sans_slash}_method_{method}','const']+variables_list].dropna()

        
        model = sm.OLS(bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'], bunching[['const']+variables_list])
        results = model.fit()
        results.params
        print(results.summary())
        
        
        sns.pairplot(data= bunching[[f'Somme_seuils_{seuils_sans_slash}_method_{method}']+variables_list])
        # p.fig.suptitle(f"Corrélation entre les différentes méthodes de mesure du bunching\naux seuils {seuils_sans_slash}, old_built_filter = {old_built_filter}")
        # p.fig.subplots_adjust(top=0.92)


    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
    

if __name__ == '__main__':
    main()
