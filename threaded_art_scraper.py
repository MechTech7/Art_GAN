import os
import requests
import bs4
from bs4 import BeautifulSoup
import json
import urllib.request
import cv2 #look into reszing images in opencv and storing them as 64x64 inputs

import concurrent.futures
import threading
import multiprocessing
import queue
import itertools

thread_local = threading.local()
img_url_queue = queue.Queue(maxsize=3600)

def get_session():
	if not hasattr(thread_local, "session"):
		thread_local.session = requests.Session()
	return thread_local.session
		

def threaded_url_list(base_url, page_number, genre):
	url = base_url + "/" + str(page_number) + "?json=2"
	global img_url_queue
	print("------------------------", page_number)
	try:
		page = get_session().get(url)
		#print(page.text)
		jayson = json.loads(page.text)
		for painting in jayson["Paintings"]:
			img_url = painting['image']
			img_url_queue.put(img_url)
			urllib.request.urlretrieve(img_url, "./impressionist_paintings/" + img_url.split("/")[-1])
			print(img_url)
	except Exception as e:
		print("Error pulling page", page_num)
	
def scrape_all(base_url, page_range):
	with concurrent.futures.ThreadPoolExecutor(10) as executor:
		executor.map(threaded_url_list, list(base_url), page_range)


if __name__ == "__main__":
	page_count = 100
	page_range = range(page_count)
	
	#Select the genre that should be scraped from this list
	genre_list = ["abstract", "advertisement", "allegorical painting", "animal painting", "architecture", "battle painting", "bijinga", "bird-and-flower painting"
    				"calligraphy", "capriccio", "caricature", "cityscape", "cloudscape", "design", "figurative", "flower painting", "genre painting",
					"graffiti", "history painting", "illustration", "installation", "interior", "jewelry", "landscape", "literary painting", "marina",
					"miniature", "mosaic", "mythological painting", "nude painting nu", "pastorale", "performance", "photo", "pin-up", "portrait"
					"poster", 
    religious painting
    sculpture
    self-portrait
    shan shui
    sketch and study
    still life
    symbolic painting
    tapestry
    tessellation
    trompe-loeil
    urushi-e
    vanitas
    veduta
    wildlife painting
    yakusha-e

	bs_url = itertools.repeat("https://www.wikiart.org/en/paintings-by-style/impressionism", page_count)
	
	scrape_all(bs_url, page_range)

	print("Print # of images scraped: ", len(list(img_url_queue.queue)))


