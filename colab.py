try:
    from google.colab import files
except:
    pass

'''
import sys
if not '/binhe/ml-ex' in sys.path:
    sys.path.append('/binhe/ml-ex')

from importlib import reload
from imp import reload
import mylibs 
try:
    reload(mylibs.colab)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
from mylibs.colab import upload,download,download_file_from_google_drive
'''

# use this to upload files
def upload():  
    try:
        uploaded = files.upload() 
        for name, data in uploaded.items():
            with open(name, 'wb') as f:
                f.write(data)
                print ('saved file', name)
    except:
        pass
# use this to download a file  
def download(path):
    try:
        files.download(path)
    except:
        pass
# !pip install "requests==2.29.0"

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)