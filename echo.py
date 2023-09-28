import logging
from aiogram import Bot, Dispatcher, executor, types
from settings import API_TOKEN, CLASSESS
import re
import pickle 
from catboost import CatBoostClassifier
from aiogram import types
from aiogram.types import CallbackQuery

# from sklearn.feature_extraction.text import TfidfVectorizer

# import numpy as np
# import os

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zа-яё #+_]')

def clean_text(text: str) -> str:
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text

model = CatBoostClassifier()
model.load_model('model_catboost.json', 'json')

vectorizer = pickle.load(open('vectorizer_catboost.pkl', 'rb'))

# model = pickle.load(open('xgb_64.pkl', "rb"))
# vectorizer = TfidfVectorizer()
# vectorizer = pickle.load(open("vectorizer_64.pkl", "rb"))
print('\nMODEL HAS SUCCESSFULLY LOADED\n')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

from aiogram import types

@dp.message_handler(commands=['start'])
async def start_prompt(message: types.Message):
     await message.answer(
        f'Это сервис поддержки пользователей услуг Click.uz \n \n {message.from_user.first_name}, напишите нам свой вопрос 👇 \n')

@dp.message_handler(content_types=['text'])
async def echo_(message: types.Message):
    """ Text answering
    """
    # await message.reply(message.text)
    qstn = vectorizer.transform([clean_text(message.text)])
    answer = model.predict(qstn)
    # print(answer)
    await message.reply(CLASSESS[answer[0, 0]])

    
if __name__ == '__main__':

    executor.start_polling(dp, skip_updates=True)
