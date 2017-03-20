from bs4 import BeautifulSoup
from functools import reduce
import requests
import nltk
from nltk.tokenize import sent_tokenize

global_uri = r'http://awoiaf.westeros.org'

def load_file_to_list(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def is_person(phrase):
  list_of_people = load_file_to_list('List_of_characters.txt')
  list_of_people = list(map(lambda x: x.strip(), list_of_people))
  return phrase in list_of_people

def is_plural(phrase):
  is_plural_word = lambda x: x[1] in ['NNS', 'NNPS']
  pos_tags = nltk.pos_tag(phrase.split())
  reducer = lambda x, y: x or is_plural_word(y)
  return reduce(reducer, pos_tags, is_plural_word(pos_tags[0]))

def extract_links_from_page(url):
  soup = bs4_from_url(url)
  a = soup.find(id="mw-content-text").find_all("li")
  b = list(filter(lambda x: not x.get('class'), a))
  c = list(filter(lambda x: "disambiguation" not in x.a.get('title'), b))
  d = list(map(lambda x: (x.a.get('title'), x.a.get('href')), c))
  next_links = soup.find_all("td", { "class" : 'mw-allpages-nav'})[0].find_all('a')
  if len(next_links) == 3:
    next_link = next_links[2].get('href')
    next_link = global_uri + next_link
  else:
    next_link = False
  return d, next_link

def bs4_from_url(url):
  result = None
  while result is None:
    try:
      result = requests.get(url)
    except:
      print("Connection timed out. Trying again.")
      pass
  return BeautifulSoup(result.content)

def get_list_of_pages(url):
  links, next_link = extract_links_from_page(url)
  while next_link:
    new_links, next_new_link = extract_links_from_page(next_link)
    next_link = next_new_link
    links += new_links
  return links

title_links = get_list_of_pages(r'http://awoiaf.westeros.org/index.php?title=Special:AllPages&from=Aegon+the+Younger')

def save_extr_links(links):
  title_file = open('title-file.txt', 'w+')
  link_file = open('link-file.txt', 'w+')
  for title, link in links:
    title_file.write(title + '\n')
    link_file.write(link + '\n')
  title_file.close()
  link_file.close()

def get_subs_from_h2(h2):
  next_sib = lambda x: x.next_sibling.next_sibling if x.next_sibling == '\n' else x.next_sibling
  clean_p = lambda x: sent_tokenize(x.text)[0]
  ele = next_sib(h2)
  all_h3s = []
  while ele and ele.name != 'h2':
    if ele.name == 'h3':
      if next_sib(ele) and next_sib(ele).name == 'p':
        all_h3s.append((ele.span.text, clean_p(next_sib(ele))))
    ele = next_sib(ele)
  return all_h3s

def lower_phrase(phrase):
  return ' '.join(list(map(lambda x: x.lower(), phrase.split())))

def make_qa_from_page(title, url):
  soup = bs4_from_url(url)
  clean_p = lambda x: sent_tokenize(x.text)[0]
  main_ques_begn = ["What is ", "Who is "]
  ques_begins = ["What is the ", "What are the "]
  qas = []
  content_text = soup.find(id="mw-content-text")
  if content_text and content_text.p:
    qas.append((main_ques_begn[is_person(title)] + title, clean_p(content_text.p)))
    all_h2s = list(filter(lambda x: x.span,soup.find_all('h2')))
    for h2 in all_h2s:
      for h3_text, ans in get_subs_from_h2(h2):
        ques = ques_begins[is_plural(h3_text)] + lower_phrase(h3_text + " " + h2.text) + " of " + title
        qas.append((ques,ans.strip()))
  return qas

def save_qas(qas):
  q_file = open('questions_file.txt', 'a+')
  a_file = open('answers_file.txt', 'a+')
  for q, a in qas:
    q_file.write(q + '\n')
    a_file.write(a + '\n')
  q_file.close()
  a_file.close()

qa_count = 0
all_qas = []
for title, link in title_links:
  qas = make_qa_from_page(title, global_uri + link)
  qa_count += len(qas)
  all_qas += qas
  print("Finished making " + str(qa_count) + " qas.")
