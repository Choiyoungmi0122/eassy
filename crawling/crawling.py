import time
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from datetime import datetime



    
source = ["https://linkareer.com/cover-letter/search?keyword=&organizationName=&page=1&role=%EC%9B%B9%20%EA%B0%9C%EB%B0%9C%EC%9E%90&sort=PASSED_AT&tab=all",
"https://linkareer.com/cover-letter/search?page=2&role=%EC%9B%B9%20%EA%B0%9C%EB%B0%9C%EC%9E%90&sort=PASSED_AT&tab=all",
"https://linkareer.com/cover-letter/search?page=3&role=%EC%9B%B9%20%EA%B0%9C%EB%B0%9C%EC%9E%90&sort=PASSED_AT&tab=all"]

class Crawler:
    def start(self):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--start-maximized')
            options.add_argument('incognito')
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            self.get_urls(driver = driver)

            source_path = "/Users/choiyoungmi/Desktop/code/essay/data/url.json"
            json_data = self.get_source(source_path)

            self.get_information(
                driver=driver, json_data = json_data
            )
            return True

        except Exception as e:
            print(e)
            return False
    
    def get_source(self, source_path):
        try:
            data = None
            with open(source_path, encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print("error : ", e)


    def get_urls(self, driver):
        
        urls = []


        for i in range(0, 3):
            driver.get(source[i])
            time.sleep(3)
            main_url_container = driver.find_element(By.XPATH, '//*[@id="__next"]/div[1]/div/div[4]/div/div/section/div[2]/div/div[3]/div[1]')

            # main_url_container = driver.find_element(By.XPATH, '//*[@id="__next"]/div[1]/div/div[4]/div/div/section[1]/div[2]/div/div[3]')
            main_url_list = main_url_container.find_element(By.CLASS_NAME, 'MuiBox-root')
            container_url_list = main_url_list.find_element(By.CSS_SELECTOR, '#__next > div.CoverLetterScreen__StyledWrapper-sc-23a97a9a-0.bpsHyY > div > div.MuiContainer-root.jss995.wrapper.MuiContainer-disableGutters > div > div > section > div.MuiBox-root.jss1098.jss1092 > div')
            url_lists = container_url_list.find_element(By.TAG_NAME, 'div')

            for li in container_url_list:
                try:
                    # div_tag = li.find_element(By.TAG_NAME, 'div')
                    a_tag = li.find_element(By.TAG_NAME, 'a')
                    href = a_tag.get_attribute('href')
                    urls.append(href)
                    print(href)
                except:
                    continue

        data = [{
            'urls': urls
        }]

        with open('/Users/choiyoungmi/Desktop/code/essay/data/url.json', 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent = 4)

    def get_information(self, driver, json_data):
        container_xpath = '//*[@id="coverLetterContent"]/main'

        informations = []
        for data in json_data:
            urls = data["urls"]

            if urls:
                for index, url in enumerate(urls):
                    print(f"{index+1}/{len(urls)}, current_url : {url}")
                    try:
                        driver.get(url)
                        WebDriverWait(driver, 10)
                        time.sleep(5)

                        all_lines = driver.find_element(By.XPATH, container_xpath).text
                        text = all_lines[:]

                        result = [text]

                        informations.append(result)
                    except Exception as e:
                        print(e)

        with open('/Users/choiyoungmi/Desktop/code/essay/data/information_webdevelop.json', 'w', encoding='utf-8') as json_file:
            json.dump(informations, json_file, ensure_ascii=False, indent = 4, default=str)

if __name__ == '__main__':
    crawler = Crawler()
    crawler.start()