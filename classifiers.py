__author__ = 'Stuart'
import urllib3
print "Welcome to Stuart's Test Classifier !"
url = raw_input("Please input the url of the data you want to classify   :")
import url_checker
if url_checker.exists(url)==True: print "Great found it !"
else: print "Sorry, %s doesn't seem to exist. Let's try again " % url




















