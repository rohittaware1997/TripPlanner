import requests
import datetime
import pickle
import redis
import json
import re

from flask import (
    Blueprint, request, jsonify
)

bp = Blueprint('categories', __name__, url_prefix='/categories')


@bp.route('/get_categories', methods=('GET', 'POST'))
def get_categories():
    r = redis.Redis('localhost')
    print('started redis server')
    redis_key = get_redis_key(request.get_json())
    print(redis_key)
    if r.get(redis_key) == None:
        res = requests.post("http://127.0.0.1:5200/get_recommadations", json = request.get_json());
        resturnRes = convertToRequiredFormat(res.json())
        serialized_data = json.dumps(resturnRes)
        r.set(redis_key, serialized_data)
        return resturnRes
    else:
        print('Found similar request previously so using cached data')
        retrieved_data = r.get(redis_key)
        deserialized_data = json.loads(retrieved_data)
        return deserialized_data

def get_redis_key(request_json):
    provience = request_json.get('province')
    price_low = str(request_json.get('low'))
    price_high = str(request_json.get('high'))
    date_start = request_json.get('begin_date')
    date_end = request_json.get('end_date')
    keys = request_json.get('cat_rating').keys()
    key_category = ''
    for i in keys:
        key_category = key_category + "#" + i 
    final_key =  provience + "#" + price_low + "#" + price_high + "#" + key_category + "#" + date_start + "#" + date_end
    return final_key

def convertToRequiredFormat(reponseBody):

    totalDays =  datetime.datetime.strptime(request.get_json().get('end_date'), '%Y-%m-%d').date().day - datetime.datetime.strptime(request.get_json().get('begin_date'), '%Y-%m-%d').date().day
    responseBodyToReturn = {}

    responseBodyToReturn.update({'categories':{}})
    responseBodyToReturn.update({'recommandations':{}})
    for i in range(totalDays+1):
        responseBodyToReturn.get('categories').update({i+1 : []})

    day = 0
    for i in range((totalDays + 1) * 8):
        place = {}
        place['category'] = clean_and_capitalize(reponseBody.get('category')[i]) 
        place['image'] = reponseBody.get('image')[i]
        place['location'] = reponseBody.get('location')[i]
        place['timeofday'] = reponseBody.get('timeofday')[i]
        place['rating'] = reponseBody.get('rating')[i]
        place['name'] = clean_and_capitalize(reponseBody.get('name')[i])
        place['price'] = reponseBody.get('price')[i]
        date = datetime.datetime.strptime(request.get_json().get('begin_date'), '%Y-%m-%d')+ datetime.timedelta(days=((int)((day/8)))); 
        place['date'] = date.strftime('%Y-%m-%d')
        place['locationToggle'] = True
        place['calendarToggle'] = True
        responseBodyToReturn.get('categories').get(((int)((day/8) + 1))).append(place)
        day += 1
    
    responseBodyToReturn.get('recommandations').update(getRecommandationsForEachDay(responseBodyToReturn, totalDays))
    return responseBodyToReturn

def getRecommandationsForEachDay(reponseBody, totalDays):
    recommdation = {}
    for i in range(totalDays+1):
        recommdation.update({i+1 : []})

    for i in range(totalDays + 1):
        dictMorning = {}
        dictEvening = {}
        for k in range(4):
            cat = reponseBody.get('categories').get(i+1)[k]
            dictMorning[cat.get('price')] = cat
        
        for k in range(4,8):
            cat = reponseBody.get('categories').get(i+1)[k]
            dictEvening[cat.get('price')] = cat

        myKeys = list(dictMorning.keys())
        myKeys.sort()
        sorted_dict_Morning = {i: dictMorning[i] for i in myKeys}

        myKeys = list(dictEvening.keys())
        myKeys.sort()
        sorted_dict_Evening = {i: dictEvening[i] for i in myKeys}

        for morning in range(2):
            value = sorted_dict_Morning.get(list(sorted_dict_Morning.keys())[morning])
            recommdation.get(i+1).append(value)

        for evening in range(2):
            value = sorted_dict_Evening.get(list(sorted_dict_Evening.keys())[evening])
            recommdation.get(i+1).append(value)


    return recommdation
        
def clean_and_capitalize(string):
    string = re.sub(r'[^\x00-\x7F]+|_', ' ', string)
    string = ' '.join(word.capitalize() for word in string.split())
    return string
