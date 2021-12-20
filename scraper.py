import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import validators

def scrape(URL):
    domain = urlparse(URL).netloc
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # Finds all "<a 'href'" elements on the html page, and thereafter removes duplicates
    url_results = soup.find_all('a', href=True)
    url_results = list(dict.fromkeys(url_results))
    url_list = []
    text_list = []

    # Finds all the "<p>" tags on the webpage and appends every line of text as... text in text_list.
    text_results = soup.find_all('p')
    for text in text_results:
        text_list.append(text.text)

    # Iterates over the urls in url_results and creates a list with only valid URL's
    # For example, sites referring to their own pages using just "/pagename" references
    # are concatenated to a correct URL. I used 'http' in the hope of catching older sites for archiving,
    # or getting an https redirect.
    for url in url_results:
        url = url['href']
        if url.startswith('/'):
            url = 'http://' + domain + url
        if validators.url(url):
            url_list.append(url)

    # In the end there is text_results with all text on one page and url_list with referenced URL's.
    # [0] is the list of text, [1] the url list
    return [text_list, url_list]

URL = "https://nl.wikipedia.org/wiki/Nederland"
#URL = "https://nos.nl"

results = scrape(URL)
for lines in results[0]:
    print(lines)
for urls in results[1]:
    print(urls)