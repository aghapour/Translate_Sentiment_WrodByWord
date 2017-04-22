
# coding: utf-8

# # Requirement:
# 
#      google-cloud-language==0.24.0
#      google-cloud-translate==0.24.0
# 
#      Install the Google Cloud SDK and authenticate with your google account.
#      gcloud beta auth application-default login

# # import

# In[309]:

import os
import glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text


# # Functions

# * Translate using google offical API

# In[310]:

from google.cloud import translate


# In[311]:

# Return the translated text to target language
def translate_text_GoogleAPI(text, target, model=translate.NMT):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target, model=model)
    return result['translatedText']


# * Translate using URL

# In[312]:

from urllib import request
from urllib.parse import quote_plus


# In[313]:

def translate(text, sourceLanguage='tr', targetLanguage='en'):
    req = request.Request("https://translate.googleapis.com/translate_a/single?client=gtx&sl={0}&tl={1}&dt=t&q={2}"
                                 .format(sourceLanguage, targetLanguage, quote_plus(str(text), encoding='utf-8')))
    req.add_header("user-agent", "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36")
    resp = request.urlopen(req)
    respText = resp.read().decode('utf-8')

    # get translated text
    translation = ''
    index = respText.find(",,\"{0}\"".format(sourceLanguage))
    if index == -1:
        startQuote = respText.find('\"')
        if startQuote != -1:
            endQuote = respText.find('\"', startQuote + 1);
            if endQuote != -1:
                translation = respText[startQuote + 1: endQuote - startQuote + 3]
                 
    return translation


# * Translate using Json

# In[314]:

try:    import json
except: import simplejson as json


# In[315]:

def translate_text_url(text, src = 'tr', to = 'en'):
  parameters = ({'langpair': '{0}|{1}'.format(src, to), 'v': '1.0' })
  translated = ''

  for text in (text[index:index + 4500] for index in range(0, len(text), 4500)):
    parameters['q'] = text
    response = json.loads(urllib.request.urlopen('http://ajax.googleapis.com/ajax/services/language/translate',
                                                 data = urllib.parse.urlencode(parameters).encode('utf-8')).read().decode('utf-8'))

    try:
      translated += response['responseData']['translatedText']
    except:
      pass

  return translated


# * Translate using Goslate

# In[316]:

# https://pythonhosted.org/goslate/
import goslate


# In[317]:

def translate_text_Goslate(text, target):
    gs = goslate.Goslate()
    return gs.translate(text, target)


# * Sentiment Analysis

# In[354]:

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(os.path.join(os.getcwd(),
                                                                                 'MyFirstGoogleCloudProject-f895d017e052.json'))


# In[345]:

from google.cloud import language, exceptions


# In[343]:

# Return score, magnitude
def sentiment_text(text):
    try:
        language_client = language.Client()
        myDocument=language_client.document_from_text(content=str(text))
        annotations=myDocument.annotate_text(include_entities=True, include_sentiment=True, include_syntax=True)
        return annotations.sentiment.score, annotations.sentiment.magnitude
    except exceptions.BadRequest as eBad:
        return 'Error', 'Error'


# # Main

# In[320]:

folderPath=os.path.join(os.getcwd(), 'ExtractedPureTurkish')


# In[321]:

myFilesList = glob.glob(os.path.join(folderPath, '*.TextGrid'))


# * Read Data to DataFrame

# In[322]:

myDataFrame = None
# myDataFrame = pd.core.frame.DataFrame


# In[323]:

for myFile in myFilesList: 
    if os.path.getsize(myFile) > 1:
#         print(os.path.split(myFile)[1])
#         myDataFrame=pd.read_csv(myFile, sep=":",header=None, error_bad_lines=False, warn_bad_lines=False)
        myDataFrame=pd.concat([myDataFrame, pd.read_csv(myFile, sep=":",header=None, error_bad_lines=False, warn_bad_lines=False)])
        


# In[324]:

myDataFrame.shape


# In[325]:

# Show the NaN in columns 2,3,4,5
(myDataFrame[~myDataFrame[2].isnull()]).shape[0] + (myDataFrame[~myDataFrame[3].isnull()]).shape[0] + \
(myDataFrame[~myDataFrame[4].isnull()]).shape[0] +(myDataFrame[~myDataFrame[5].isnull()]).shape[0]


# * Vecorizing

# In[326]:

stopWords=['', ' ']


# In[327]:

vectorizer = None
vectorizer = text.CountVectorizer(stop_words=stopWords)
newData = vectorizer.fit_transform(myDataFrame[myDataFrame[0]=='m'][1].values.astype('U'))
print(newData.shape)
maleWordList = pd.DataFrame(vectorizer.get_feature_names())


# In[328]:

vectorizer = None
vectorizer = text.CountVectorizer(stop_words=stopWords)
newData = vectorizer.fit_transform(myDataFrame[myDataFrame[0]=='f'][1].values.astype('U'))
print(newData.shape)
femaleWordList = pd.DataFrame(vectorizer.get_feature_names())


# * Save DataFrame to file

# In[329]:

maleWordList.to_csv('maleWordList.csv', header=None, encoding='utf-8')


# In[330]:

femaleWordList.to_csv('femaleWroedList.csv', header=None, encoding='utf-8')


# * transalte male and female Dataframe and Save them to file as a CSV file

# In[331]:

maleWordListEn = pd.DataFrame([translate(text=X[0], sourceLanguage='tr',targetLanguage='en') for X in maleWordList.values])


# In[333]:

maleWordListEn.to_csv('maleWordListEn.csv', header=None, encoding='utf-8')


# In[332]:

femaleWordListEn = pd.DataFrame([translate(text=X[0], sourceLanguage='tr',targetLanguage='en') for X in femaleWordList.values])


# In[334]:

femaleWordListEn.to_csv('femaleWordListEn.csv', header=None, encoding='utf-8')


# * sentiment male and female Dataframe and Save them as CSV file

# In[355]:

maleWordListEnSent = pd.DataFrame([(sentiment_text(text=X[0])[0], print(X)) for X in maleWordListEn.values])


# In[ ]:

maleWordListEnSent.to_csv('maleWordListEnSent.csv', header=None, encoding='utf-8')


# In[ ]:

femaleWordListEnSent = pd.DataFrame([(sentiment_text(text=X[0])[0], prin(X)) for X in femaleWordListEn.values])


# In[ ]:

femaleWordListEnSent.to_csv('femaleWordListEnSent.csv', header=None, encoding='utf-8')


# In[ ]:



