import tensorflow as tf
import numpy as np
import re
import nltk
import sys
import time
import math
import random
import os

## Getting and Cleaning the Data

book_path = 'gottext.txt'
read_data = ""

print('Preparing data...')
with open(book_path) as book:
	read_data = book.read()

quotes = re.findall('“([^”]*)”', read_data)
quotes = quotes[:500]
