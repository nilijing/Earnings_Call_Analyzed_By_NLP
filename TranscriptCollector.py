# -*- coding: utf-8 -*-
# ScrapeData_EA.py
# scrapes earning call data from Seeking Alpha

import requests
import time
from bs4 import BeautifulSoup

import os
from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.interaction import KEY
from selenium.webdriver.common import keys
import json

def create_text(filename,mtext): 
  file_path=filename + '.txt'
  file = open(file_path,'w') 
  file.write(mtext)
  file.close

def grab_page(url,CompanyCount,StockTicker):
    #Chrome Session
    #driver=webdriver.Chrome()   
    driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(100)
    print("attempting to grab page: " + url)
    # page = requests.get(url)
    # page_html = page.text
    #soup = BeautifulSoup(page_html, 'html.parser')
  
    action = ActionChains(driver)  
    action.move_by_offset(8,1)
    action.move_by_offset(6,1)
    action.move_by_offset(4,1)
    action.move_by_offset(2,1)
    action.move_by_offset(1,1)
    action.move_by_offset(1,2)
    action.move_by_offset(1,4)
    action.move_by_offset(1,6)
    action.move_by_offset(1,8)
    action.release()
    action.perform()   
               
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    meta = soup.find('script', type='application/ld+json')
    
    if (type(meta)==type(None)):
        print("skipping this link, no content here")
        return
    else:
        mtext = meta.string
        start_tag=mtext.find('articleBody')
        end_tag=mtext.find('(',start_tag)
        quarter_tag=mtext.find('Earnings',start_tag)
        company_name=str(StockTicker)+"_"+str(CompanyCount)
        if(end_tag-start_tag<150):
            text_name=mtext[(start_tag+14):(end_tag-1)]+ "_"+ mtext[(quarter_tag-8):(quarter_tag-1)]
        
        filename ="/content/drive/MyDrive/Earnings_call_NLP/AlphaStreet/S&P 500"        

        path = os.path.join(filename, StockTicker)  
        try:
            os.mkdir(path)
        except OSError as error:  
            print("Folder exists")  

        filename = path+"/"+ text_name
        if(os.path.isfile(filename+".txt")):
            filename = path+"/"+ text_name+"_"+str(CompanyCount)
        else:
            print('filename:',text_name)  

        #file = open(filename.lower() + ".txt", 'w')
        #file.write(mtext)
        #file.close
        create_text(filename,mtext)
        print(text_name.lower()+ " sucessfully saved")
    driver.quit()
    

def grab_audio(url,Quarter,StockTicker):
    #Chrome Session
    #driver=webdriver.Chrome()    
    driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(100)
    print("attempting to get earning call audio for: " + StockTicker +" "+ Quarter)
    driver.find_element_by_id('playButton').click()
    #Not closing driver, will have to manually close the audio in our exmaple.
    return driver

def process_list_page(StockTicker):    
    #driver=webdriver.Chrome()    
    driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    origin_page = "https://alphastreet.com/earnings/earnings-call-transcripts?ticker="+str(StockTicker)
    print("getting page " + origin_page)
    driver.get(origin_page)
    driver.implicitly_wait(100)   
   
    # page = requests.get(origin_page)
    # page_html = page.text
    #print(page_html)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    alist = soup.find_all("div",{'class':'product-trans-button'})
    #Collect data for last 2 years
    transcript_count=len(alist) if 12 > len(alist) else 12 
    for i in range(0,len(alist)):
        url_ending = alist[i].find_all("a")[1].attrs['href']
        if(url_ending.split("/")[2]=="earnings-call-transcripts"):
            url = "https://alphastreet.com/" + url_ending
            grab_page(url,i,StockTicker)
            time.sleep(.5)
            
    driver.quit()


def process_Call(StockTicker,Quarter):   
    #driver=webdriver.Chrome() 
    driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    main_driver=""
    origin_page = "https://alphastreet.com/earnings/earnings-call-transcripts?ticker="+str(StockTicker)
    print("getting page " + origin_page)
    driver.get(origin_page)
    driver.implicitly_wait(100)
     
    # page = requests.get(origin_page)
    # page_html = page.text
    #print(page_html)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    alist = soup.find_all("div",{'class':'product-trans-button'})
    qlist=soup.find_all("div",{'class':'as-productfeed-item-title'})
    #Collect data for last 2 years
    transcript_count=len(alist) if 12 > len(alist) else 12 
    for i in range(0,len(alist)):
        url_ending = alist[i].find_all("a")[0].attrs['href']
        quarter_details=qlist[i].find_all("h3")[0].text
        Start_tag=qlist[i].find_all("h3")[0].text.find('Earnings')
        if((url_ending.split("/")[2]=="earnings-calls") and (qlist[i].find_all("h3")[0].text[Start_tag-8:Start_tag-1]==Quarter)):
             url = "https://alphastreet.com/" + url_ending
             main_driver=grab_audio(url,Quarter,StockTicker)
             time.sleep(.5)
            
    driver.quit()
    return main_driver


def Get_Transcripts(Stock_Tickers):
  for i in range(len(Stock_Tickers)):
        print("******Getting Transcripts for:"+ str(Stock_Tickers['Symbol'][i]))
        try:
           process_list_page(Stock_Tickers['Symbol'][i])
        except:
          print("Error in " +str(Stock_Tickers['Symbol'][i])) 
  
  print('All targeted transcripts done.') 


def Get_EarningCall(Stock_Tickers,Quarter):
    main_driver=""
    for i in range(len(Stock_Tickers)):
        print("******Getting Earning Call Details for:"+ str(Stock_Tickers['Symbol'][i]))
        try:
           main_driver=process_Call(Stock_Tickers['Symbol'][i],Quarter)
        except:
                print("Error in " +str(Stock_Tickers['Symbol'][i]))  
    return main_driver