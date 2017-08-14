import praw
from robobrowser import RoboBrowser
from bs4 import BeautifulSoup
import re

import utils.config as config

def get_submissions(query,subreddit,limit=3):
	return subreddit.search(query, syntax='lucene',limit=limit)

def make_query(site,topic):
	return 'site:'+site+'.com'+' '+topic

def get_subreddit(name):
	reddit = praw.Reddit(client_id=config.client_id,
                     client_secret=config.client_secret,
                     user_agent=config.user_agent)

	return reddit.subreddit(name)

def fetch_sources(topic, subreddit_name = 'all',
				  quantity = 1,sites = ['cnn','foxnews','thehill','usatoday',"breitbart"]):
	
	subreddit = get_subreddit(subreddit_name)
	res = []

	for site in sites:
		query = make_query(site,topic)
		submissions = get_submissions(query,subreddit,quantity)
		
		for submission in submissions:
			res.append({'site':site,'url':submission.url})

	return res

def get_articles(topic, subreddit_name = 'all',
				  quantity = 1,sites = ['cnn','foxnews','thehill','usatoday',"breitbart"]):

	browser = RoboBrowser()
	sources = fetch_sources(topic,subreddit_name,quantity,sites)

	for source in sources:
		url = source['url']
		site = source['site']
		browser.open(url)

		#clean script tags
		scripts = browser.find_all(['script', 'style'])
		for script in scripts:
			script.decompose()   


		#get content

		#try itemprop=
		content = browser.parsed.find(itemprop = "articleBody")
		if content: 
			content = content.get_text()

		#if that doesn't work, try <article> tag
		if not content:
			content = browser.parsed.find("article")
			if content: 
				content = content.get_text()

		#still didn't work? grab all the p tags
		if not content:
			content = browser.parsed.find_all("p")
			content = BeautifulSoup('\n\n'.join(str(s) for s in content), 'html.parser')
			if content: 
				content = content.get_text()

		#giveup
		if not content:
			content = browser.parsed.find("body")
			if content: 
				content = content.get_text()

		#modify source
		sentences = re.split('[?.!]', str(content))
		source["sentences"] = sentences
		

	return sources










