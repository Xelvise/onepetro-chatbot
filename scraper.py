import os, time, json
from undetected_chromedriver import Chrome as ChromeDriver
# from undetected_chromedriver.options import ChromeOptions
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium_recaptcha_solver import RecaptchaSolver, StandardDelayConfig
# from selenium_recaptcha_solver.exceptions import RecaptchaException
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from logger import logging
from bs4 import BeautifulSoup


class Bot(ChromeDriver):
    def __init__(self):
        # Initialize the inherited Chrome webdriver
        super(Bot, self).__init__(
            driver_executable_path=ChromeDriverManager().install(),   # downloads Chrome driver and returns the path, if driver already exists
            # Replace None with path to Chrome user-data directory (Search 'chrome://version' on your browser to find yours)
            user_data_dir=None
        )
        self.implicitly_wait(5)

    # Uncomment the code block below, if you'd want the browser open even after the script is executed
    # def __del__(self):
    #     try:
    #         self.service.process.kill()
    #     except:
    #         pass
    
    # Uncomment the code block below, if you require User authentication

    # def solve_captcha(self):
    #     # Solve the reCAPTCHA
    #     solver = RecaptchaSolver(driver=self, delay_config=StandardDelayConfig(6,10))
    #     recaptcha_iframe = self.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
    #     try:
    #         solver.click_recaptcha_v2(iframe=recaptcha_iframe)
    #     except TimeoutException as e:
    #         self.solve_captcha()

    # def user_auth(self, username:str, password:str):
    #     ''' Authenticate the user with the given username and password\n 
    #         Returns the login status and userid (if logged-in)'''
    #     self.get("https://onepetro.org/my-account/library")
    #     if self.current_url != "https://onepetro.org/my-account/library":       # Check if the user is logged in
    #         time.sleep(2)
    #         username_field = self.find_element(By.CSS_SELECTOR, "#user_LoginForm")
    #         username_field.clear()
    #         username_field.send_keys(username + Keys.TAB)
    #         time.sleep(2)
    #         password_field = self.find_element(By.CSS_SELECTOR, "#pass_LoginForm")
    #         password_field.clear()
    #         password_field.send_keys(password)

    #         try:
    #             WebDriverWait(self, 10).until(
    #                 EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))    # Wait for the reCAPTCHA to load
    #         except TimeoutException:
    #             self.switch_to.default_content()
    #             self.find_element(By.XPATH, '//button[text()="Sign In"]').click()   # Click the Sign In button
    #         else:
    #             try:
    #                 self.solve_captcha()    # Solve the reCAPTCHA
    #             except:
    #                 self.captcha_status = 'failed'
    #                 self.get_screenshot_as_file(os.path.join(os.getcwd(),'logs','screenshots','recaptcha.png'))
    #                 pass    # If the reCAPTCHA solving fails, we'd have to resort to a paid provider   
    #             else:
    #                 self.captcha_status = 'solved'
    #                 self.switch_to.default_content()
    #                 self.find_element(By.XPATH, '//button[text()="Sign In"]').click()   # Click the Sign In button
            

    def select_papers(self, url):
        '''Selects all papers published on a given day and returns their paper headers as a list of dictionaries
        '''
        self.get(url)
        collection = list()
        papers = self.find_elements(By.CSS_SELECTOR, '#resourceTypeList-ConferenceVolumeBrowse_Volume > div > section > div.content.al-article-list-group > div')

        for paper in papers:
            title = paper.find_element(By.CSS_SELECTOR, "div > h5 > a").text    # extract title of paper
            authors = paper.find_elements(By.CSS_SELECTOR, 'div > div.al-authors-list > span.wi-fullname.brand-fg a')    # extract list of authors per paper
            authors = ', '.join([author.text for author in authors])
            presentation_date = paper.find_element(By.CSS_SELECTOR, 'div > div.al-cite-description').text   # extract presentation date
            presentation_date = presentation_date.replace('Paper presented at the SPE Nigeria Annual International Conference and Exhibition, Lagos, Nigeria, ', '').split('.')[0]
            ref_link = paper.find_element(By.CSS_SELECTOR, 'div > div.al-cite-description > span.citation-doi a').get_attribute('href')     # extract reference link
            collection.append({'title': title, 'authors': [authors], 'presentation_date': presentation_date, 'reference_link': ref_link})

        # json.dump(collection, open('search_results.json', 'w', encoding='utf-8'), indent=4)    # Uncomment this line if you'd like to save the page headers to a json file
        return collection


    def scrape_and_save_paper(self, paper_headers:list[dict], year:str) -> None:
        ''' `paper_headers`: list of dictionaries containing the paper headers\n
            `year`: year to be used as subdirectory for storing the text files. E.g: `2022`,`2023`\n
        '''
        try:
            os.makedirs(f'files/{year}', exist_ok=True)     # Create a subdirectory to store the text files, if it doesn't exist
            for paper in paper_headers:
                subdirectory = year.strip()    # strips any whitespaces in the subdirectory
                title = paper['title'].replace(':','-')  # Replaces colon with hyphen in the title
                filename = "".join(c for c in title if c.isalnum() or c in ' -_').strip()    # Create a filename from the paper title
                file_path = os.path.join('files', subdirectory, f'{filename}.txt')     # Create a file path for the paper 

                if os.path.isfile(file_path):   # Using the file_path, checks if the file exists
                    continue    # skips to the next paper, if the file already exists
                
                # If the file doesn't exist in file_path, scrape the content of the paper
                self.get(paper['reference_link'])
                content = BeautifulSoup(self.page_source, 'lxml')

                with open(file_path, 'w', encoding='utf-8') as f:     # open and append content to it
                    # Write the metadata to the text file
                    f.write("----- METADATA START -----\n")
                    f.write(f"Title: {paper['title']}\n")
                    f.write(f"Authors: {', '.join(paper['authors'])}\n")
                    f.write(f"Publication Date: {paper['presentation_date']}\n")
                    f.write(f"Reference Link: {paper['reference_link']}\n")
                    f.write("----- METADATA END -----\n\n")

                    # Write the content to the text file
                    f.write(content.find('div', attrs={'data-widgetname':"ArticleFulltext"}).text)
        except Exception as e:
            logging.info(f"An error occurred: {e}")
            print(e)

DAY2 = "https://onepetro.org/SPENAIC/23NAIC/conference/2-23NAIC"    # url of SPENAIC conference DAY2 2023
DAY3 = "https://onepetro.org/SPENAIC/22NAIC/conference/3-23NAIC"    # url of SPENAIC conference DAY3 2023

if __name__ == '__main__':
    bot = Bot()
    # bot.user_auth('Joy Ugoyah', '#YXx3ievQnPkH5S')      # Uncomment this line, only if User auth is required
    articles = bot.select_papers(url=DAY2)      # Returns a list of dictionaries with each containing a paper header
    bot.scrape_and_save_paper(articles, year='2023')     # Using the paper headers, scrapes and saves the papers to folder '2023'

