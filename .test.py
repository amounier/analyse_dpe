#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:58:40 2026

@author: amounier
"""

import time
import os
from urllib.request import urlopen, Request
import json
import jsondiff
import subprocess

from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# def download_dpe_details_requests(dpe_id, force=False):
#     """
#     Ne marche pas. 403 forbidden ou 400 bad request (protection par captcha)

#     """
#     output_folder_dpe_details = os.path.join('data', 'DPE', 'XML')
#     os.makedirs(output_folder_dpe_details, exist_ok=True)
    
#     if '{}.json'.format(dpe_id) in os.listdir(output_folder_dpe_details) or force:
#         return

#     else:
#         user_agents_list = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
#                             "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
#                             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
#                             "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
#                             "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:134.0) Gecko/20100101 Firefox/134.0",]
        
#         HEADERS = {"User-Agent": rd.choice(user_agents_list),
#                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
#                    "Accept-Language": "en-US,en;q=0.5",
#                    "Accept-Encoding": "gzip, deflate",
#                    "Connection": "keep-alive",
#                    "Upgrade-Insecure-Requests": "1",
#                    "Sec-Fetch-Dest": "document",
#                    "Sec-Fetch-Mode": "navigate",
#                    "Sec-Fetch-Site": "none",
#                    "Sec-Fetch-User": "?1",
#                    "Cache-Control": "max-age=0",}
        
#         # dls = f"https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines?numero_dpe_eq={dpe_id}" 
#         dls = f'https://observatoire-dpe-audit.ademe.fr/pub/dpe/{dpe_id}/xml'
#         print(dls)
#         req = Request(dls,headers=HEADERS)
#         content = urlopen(req)
        
#         # with open(os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id)), 'wb') as output:
#         with open(os.path.join('data','DPE','XML','{}.xml'.format(dpe_id)), 'wb') as output:
#             output.write(content.read())
            
#     return



def download_dpe_details_selenium(dpe_id, force=False, timeout = 20, sleep_time=0.5):
    destination_folder = os.path.join(os.getcwd(),'data', 'DPE', 'XML')
    os.makedirs(destination_folder, exist_ok=True)
    success = True
    
    if f'{dpe_id}.xml' not in os.listdir(destination_folder) or force:
        success = False
        
        # téléchargement du fichier xml
        try:
            _ = ChromeDriverManager().install()
            options = webdriver.ChromeOptions() # https://peter.sh/experiments/cgeadhromium-command-line-switches/ 
    
            driver = webdriver.Chrome(options=options)
    
            driver.get("https://observatoire-dpe-audit.ademe.fr/accueil")
            wait = WebDriverWait(driver,timeout)
            
            selector_cookies = '#tarteaucitronAllDenied2'
            wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value=selector_cookies))
            driver.find_element(by=By.CSS_SELECTOR, value=selector_cookies).click()
            
            time.sleep(sleep_time)
            
            selector_dpe = '#bienvenue > div > div.actions-widgets > app-widget-trouver-dpe > div.search-dpe > form > app-input-text-clearable > div > input'
            wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value=selector_dpe))
            driver.find_element(by=By.CSS_SELECTOR, value=selector_dpe).send_keys(dpe_id+Keys.ENTER)
            
            time.sleep(sleep_time)
            
            selector_download = '#dd-dl-detail-dpe'
            wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value=selector_download))
            driver.find_element(by=By.CSS_SELECTOR, value=selector_download).click()
            
            time.sleep(sleep_time)
            
            selector_xml = 'body > app-root > main > app-page-public-detail-dpe > app-detail-dpe > app-panel > div.header > div.header-right > app-dropdown > div > div > button:nth-child(2)'
            wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value=selector_xml))
            driver.find_element(by=By.CSS_SELECTOR, value=selector_xml).click()
            
            # récupération du nom du dossier téléchargement
            result = subprocess.run(
                ["xdg-user-dir", "DOWNLOAD"],
                capture_output=True, text=True
            )
            download_folder = result.stdout.strip()
            
            time.sleep(sleep_time*3)
            
            if f'{dpe_id}.xml' in os.listdir(download_folder):
                success = True
                driver.quit()
            
            if not success:
                selector_error = 'body > ngb-modal-window > div > div > app-modal-error > app-modal-base > div.modal-footer > app-action-link > button'
                try: 
                    wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value=selector_error))
                    driver.find_element(by=By.CSS_SELECTOR, value=selector_error).click()
                    driver.quit()
                except (NoSuchElementException, TimeoutException):
                    if f'{dpe_id}.xml' in os.listdir(download_folder):
                        success = True
                    driver.quit()
            
            # déplacement du fichier téléchargé
            if success:
                origin_path = os.path.join(download_folder,f'{dpe_id}.xml')
                subprocess.call(f"mv {origin_path} {destination_folder}", shell=True)
                
        except TimeoutException:
            success = False
        
    return success



def download_dpe_details(dpe_id, force=False):
    """
    Téléchargement des fichiers de sorties des DPE (au format XML)

    Parameters
    ----------
    dpe_id : str
        identifiant du dpe.
    force : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    output_folder_dpe_details = os.path.join('data', 'DPE', 'JSON')
    os.makedirs(output_folder_dpe_details, exist_ok=True)
    
    if '{}.json'.format(dpe_id) in os.listdir(output_folder_dpe_details) or force:
        return

    else:
        dls = f"https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines?numero_dpe_eq={dpe_id}" 
        req = Request(dls)
        content = urlopen(req)

        # with open(os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id)), 'wb') as output:
        with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id)), 'wb') as output:
            output.write(content.read())
    return


def compare_dpe_data(dpe_id1,dpe_id2):
    with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id1)), 'r') as f:
        json_dpe1 = json.load(f)
    with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id2)), 'r') as f:
        json_dpe2 = json.load(f)
    dpe_diff = jsondiff.JsonDiffer().diff(a=json_dpe1, b=json_dpe2)
    return dpe_diff


def main():
    tic = time.time()
    
    dpe_id1 = '2375E2162413S'
    # dpe_id1 = '2275E2882995J'
    # download_dpe_details(dpe_id1)
    
    # dpe_id1 = '2375E2258099Y'
    # dpe_id1 = '2275E2946357J'
    # download_dpe_details(dpe_id2)
    
    # print(compare_dpe_data(dpe_id2, dpe_id1)) 
    success = False
    nb_try = 0
    while not success:
        success = download_dpe_details_selenium(dpe_id1,force=True,timeout=20)
        nb_try += 1
        print(nb_try)
        
    
    # print('Réussite :', success)
    
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()