#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:05:21 2026

@author: audrey
"""

import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import statsmodels.api as sm

from administrative import  list_dep_code, Departement
from distribution import methods_dict, calcul_bunching_france, cut_france_bunching, get_nb_dpe
from utils import etiquette_ep_seuils



def tension_immob_dep():
    """
    Détermination des proportions de logements dans chacune des 5 zones du zonage ABC pour chaque département.
    Rappel : 
    Les 5 zones sont désignées par ordre de déséquilibre décroissant :
        Zone A bis : Paris et sa proche banlieue avec une demande locative extrêmement forte.
        Zone A : grandes agglomérations où la demande est élevée (Lyon, Lille, Marseille, Montpellier, etc.).
        Zone B1 : villes moyennes dynamiques et zones frontalières ou littorales.
        Zone B2 : petites villes ou communes où la tension est modérée.
        Zone C : zones détendues où l’offre dépasse la demande.

    Parameters
    ----------
    path_tension : str
        chemin vers le fichier Excel des tensions locatives par commune.
    path_logements : str
        chemin vers le fichier Excel du nombre de logements par commune.

    Returns
    -------
    df_tension_immob_dep : pandas DataFrame
        Part des logements du département dans chacune des 5 zones 
        Colonnes :  dep_code (index)  |  total_logements  |  part_A = part de logements du département en zone A  |  part_Abis  |  part_B1  |  part_B2  |  part_C  
    """
    # chemin vers le fichier Excel des tensions locatives par commune
    path_tension = os.path.join('data','INSEE','Liste ensemble des communes - Zonage ABC 5 septembre 2025.xlsx')
    
    # chemin vers le fichier Excel du nombre de logements par commune
    path_logements =  os.path.join('data','INSEE','logement-2022.xlsx')
    
    # Import du zonage de la tension locative par commune
    df_zonage_ABC = pd.read_excel(path_tension, usecols=['CODGEO', 'DEP',  'Zonage en vigueur depuis le 5 septembre 2025'])
    df_zonage_ABC = df_zonage_ABC[~df_zonage_ABC['DEP'].isin(['971', '972', '973', '974', '975', '976'])] 
     
    # Import du nombre de logements par commune
    df_logements = pd.read_excel(path_logements, names=['CODGEO', 'LIBGEO', 'LOGEMENTS_2022'], na_values = 'N/A - résultat non disponible', skiprows=3)
    
    # Jointure des deux dataframes
    df_communes = pd.merge(df_zonage_ABC, df_logements, on='CODGEO') #, 'LIBGEO'])
    df_communes.set_index('CODGEO', inplace=True)
    df_communes = df_communes[~df_communes['DEP'].isin(['971', '972', '973', '974', '975', '976'])] # on étudie seulement la France hexagonale


    # Création du df des tensions immobilières par département
    df_tension_immob_dep = pd.DataFrame(index=list_dep_code) 
    df_tension_immob_dep['departement'] = [Departement(f'{n}') for n in df_tension_immob_dep.index]
    
    # Colonne nombre total de logements par département
    df_tension_immob_dep['total_logements'] = df_communes.groupby('DEP')['LOGEMENTS_2022'].sum()
    
    # Colonnes nombre de logements dans chaque zone par département
    logements_par_zone = df_communes.groupby(['DEP', 'Zonage en vigueur depuis le 5 septembre 2025'])['LOGEMENTS_2022'].sum()
    logements_par_zone = logements_par_zone.unstack(fill_value=0)  # unstack transforme les zones ABC en colonnes, et fill_value remplace les valeurs manquantes par 0
    
    # Colonnes parts des logements dans chaque zone par département
    parts_par_zone = logements_par_zone.div(df_tension_immob_dep['total_logements'], axis=0)
    parts_par_zone.columns = [f'part_{col}' for col in parts_par_zone.columns] # on renomme les colonnes
        
    df_tension_immob_dep = df_tension_immob_dep.join(parts_par_zone)
    
    return df_tension_immob_dep
    
    

def get_nb_diagnostiqueur_dep(force = False):
    # Nombre de diagnostiqueurs par département
    save_name = 'annuaire-diagnostiqueurs-immobiliers_light.csv'
    if save_name not in os.listdir(os.path.join('data','MTE')) or force:
        data = pd.read_csv(os.path.join('data','MTE','annuaire-diagnostiqueurs-immobiliers.csv'),sep=';')
        data = data[data['Type de certificat'].str.contains('DPE')]
        data['CP'] = [f"{e:05d}" for e in data.CP]
        data.drop_duplicates(subset = ["N° de certificat", "Organisme", "CP"], inplace = True) # beaucoup de diagnostiqueurs sont en doublons
        data = data[~data.CP.str.startswith('97')] # uniquement territoire hexagonal
        data = data[~data.CP.str.startswith('20')] # hors corse (2A et 2B agrégé)
        data['dep_code'] = [Departement(e[:2]).code for e in data.CP]
        
        data_count = data.groupby('dep_code')[['Organisme']].count()
        data_count = data_count.rename(columns={'Organisme':'nb_diagnostiqueurs_dep'})
        data_count.to_csv(os.path.join('data','MTE',save_name))
        
    data_count = pd.read_csv(os.path.join('data','MTE',save_name)).set_index('dep_code')
    data_count.index = [Departement(e).code for e in data_count.index]
    return data_count
    


def get_nb_rp_loc(force = False):
    # Nombre de résidences principales occupées par des locataires
    save_name = 'base-cc-logement-2022_light.csv'
    if save_name not in os.listdir(os.path.join('data','INSEE')) or force:
        df_nb_rp_loc = pd.read_excel(os.path.join('data','INSEE','base-cc-logement-2022.xlsx'), usecols=['CODGEO', 'P22_RP_LOC','P22_RP'], skiprows=5)  # names = ['CODGEO', 'nb_rp_loc']
        df_nb_rp_loc.set_index('CODGEO', inplace=True)
        df_nb_rp_loc = df_nb_rp_loc[~df_nb_rp_loc.index.str.startswith('97')] # uniquement territoire hexagonal
        df_nb_rp_loc = df_nb_rp_loc[~df_nb_rp_loc.index.str.startswith('20')] # hors corse (2A et 2B aggrégé)
        df_nb_rp_loc['dep_code'] = [Departement(e[:2]).code for e in df_nb_rp_loc.index]
        df_nb_rp_loc = df_nb_rp_loc.groupby('dep_code')[['P22_RP_LOC','P22_RP']].sum()
        df_nb_rp_loc['ratio_RP_loc'] = df_nb_rp_loc.P22_RP_LOC/df_nb_rp_loc.P22_RP
        df_nb_rp_loc.to_csv(os.path.join('data','INSEE',save_name))
    
    df_nb_rp_loc = pd.read_csv(os.path.join('data','INSEE',save_name)).set_index('dep_code')
    return df_nb_rp_loc[['ratio_RP_loc']]
    

    
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
        


    # ORDINARY LEAST SQUARE MODEL + PAIRPLOT ENTRE BUNCHING (SOMME SUR PLUSIEURS SEUILS) ET AUTRES VARIABLES DÉPARTEMENTALES AU CHOIX
    
    
    if True :
        
        # Fixation des paramètres de mesure du bunching
        
        old_built_filter = True

        # method='diff_simple'
        method = 'diff_beta'
        # method='diff_moyenne'
        
        itv_bunching=10
        window_size=50
        seuils = ['D/E', 'E/F', 'F/G']
        
        
        # Fixation des variables à prendre en compte pour l'analyse économétrique (modèle OLS)
        
        # variables_list = ['part_A','part_Abis','part_B1','part_B2','part_C']
        # variables_list = ['part_A','part_C','zcl_Tref','total_logements','nb_diagnostiqueurs_dep']
        variables_list = ['nb_diagnostiqueurs_dep', 'part_C', 'ratio_RP_loc', 'zcl_DH19']
        
        
        france_bunching = calcul_bunching_france(output_folder, method, itv_bunching, window_size, old_built_filter, max_xlim = 600, verbose=False, force=False)
        france_bunching_cut, seuils_sans_slash = cut_france_bunching(france_bunching, seuils)
        
        # Initialisation d'un DataFrame du bunching (somme sur seuils)
        bunching = pd.DataFrame(index=list_dep_code) 
        bunching[f'Bunching {methods_dict[method]}'] = france_bunching_cut.sum(axis=1) # on somme sur les lignes des colonnes conservées    
        
        
        # Vecteur des paramètres
        
        variables = tension_immob_dep()
        
            # ajout du log du nombre de logements
        variables['log_total_logements'] = np.log(variables.total_logements)
        
            # ajout du nombre de DPE
        nb_dpe = get_nb_dpe(old_built_filter,list(etiquette_ep_seuils.keys()))
        nb_dpe = {k.code:v for k,v in nb_dpe.items()}
        nb_dpe_df = pd.DataFrame().from_dict({'dep_code':list(nb_dpe.keys()),'nb_dpe_dep':list(nb_dpe.values())}).set_index('dep_code')
        variables = variables.join(nb_dpe_df)
        
            # ajout du nombre et de la densité en diagnostiqueurs
        variables = variables.join(get_nb_diagnostiqueur_dep())
        variables['density_diagnostiqueurs_dep'] = variables.nb_diagnostiqueurs_dep/variables.total_logements

            # ajout des variables liées au zonage climatique 
        variables["zcl"] = [Departement(e).climat for e in variables.index] 
        variables["zcl_Tref"] = [Departement(e).climat_Text_ref for e in variables.index] 
        variables["zcl_DH19"] = [Departement(e).climat_DH19_ref for e in variables.index]
        variables['zcl_H3'] = (variables.zcl == 'H3').map(int)
        
            # ajout du nombre de résidence principale en location
        variables = variables.join(get_nb_rp_loc()) 
        
            # constante
        variables = sm.add_constant(variables)
        
        
        
        bunching = bunching.join(variables)
        bunching = bunching[[f'Bunching {methods_dict[method]}','const']+variables_list].dropna()
        
        # normalisation des variables entre 0 et 1
        for c in bunching.columns:
            if c == 'const':
                continue
            bunching[c] = (bunching[c]-bunching[c].min())/(bunching[c].max()-bunching[c].min())
        
            
        model = sm.OLS(bunching[f'Bunching {methods_dict[method]}'], bunching[['const']+variables_list])
        results = model.fit()
        results.params
        print(results.summary())
        print(results.summary().as_latex())
        
        
        p = sns.pairplot(data= bunching[[f'Bunching {methods_dict[method]}']+variables_list])
        # p.fig.suptitle(f"Corrélation entre le bunching méthode {method} et d'autres variables, old_built_filter = {old_built_filter}")
        # p.fig.subplots_adjust(top=0.96)
        
        if True :
            if old_built_filter:
                save_path = os.path.join(output_folder,f'Pairplot_correlation_bunching_methode_{method}_{variables_list}_old_built.png') 
            else:
                save_path = os.path.join(output_folder,f'Pairplot_correlation_bunching_methode_{method}_{variables_list}.png') 
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()
        
        

    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    

if __name__ == '__main__':
    main()
