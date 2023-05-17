# CSE635_NLP_Project
##Topic-Based Empathic Chatbot

## Setup

```
pip install -r requirement.txt
python -m spacy download en_core_web_lg
```
## Load IR knowledge base
```
#Download SimpleWiki dump from WikiMedia website and place in root directory
python wikibot/wikibot_loader.py
```

## How to Run
```
streamlit run app.py
```
