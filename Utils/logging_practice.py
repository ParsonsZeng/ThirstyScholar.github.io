#!/bin/python3

"""
Adapt from:
1. https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
"""

import logging

# if comment out, won't display message on screen
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info('Program starts')
logger.debug('This is a debug message')

# Create a file handler
handler = logging.FileHandler('my_first_log.log')
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

logger.info('Hello world!')
