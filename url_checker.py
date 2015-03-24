__author__ = 'Stuart'
import requests
import urllib3
import requests.packages.urllib3.contrib.pyopenssl
requests.packages.urllib3.contrib.pyopenssl.inject_into_urllib3()

def exists(url):
    r = requests.head(url)
    return r.status_code == requests.codes.ok


