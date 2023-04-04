# Quant Trading
Quantitative analysis for trading. 

The aim is to help a trader to backtest strategies and select good opportunities. 
The database utilized on this project is from alphavantage. For documentation check: https://www.alphavantage.co/documentation/.  

## Installation
It's recommend to have a virtual environment folder.
To create one:
```
$ python -m venv env
```
And activate it:
```
# On windows
$ .\env\Scripts\activate
# On Mac or Linux
$ source env/bin/activate
```

Then install all the requirements:

```
$ pip install -r requirements.txt
```
To start a new jupyter-lab:
```
$ python -m ipykernel install --name=env
```

## Please collaborate!
Develop is meant to include new strategies to the strategies.py file as well as implement improvements to the trader code. 

Any improvemnt spotted out should be posted as a new issue. 

The gui-app is on its initial stage and has little done. I am currently using streamlit python package for that. 