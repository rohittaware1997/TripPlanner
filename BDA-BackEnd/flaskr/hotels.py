import requests

from flask import (
    Blueprint, request, jsonify
)

bp = Blueprint('hotels', __name__, url_prefix='/hotels')

# @bp.route('/get_hotel_ammenities')
# def get_hotel_ammenities():
#     res = requests.get("http://127.0.0.1:5200/get_ammenities");
#     return res.json()

@bp.route('/get_hotel_recommandations/', methods=['POST'])
def get_hotel_recommandations():

    res = requests.post("http://127.0.0.1:5200/get_hotel_recommandations", json=request.get_json());
    return convertToRequiredFormat(res.json())
    
def convertToRequiredFormat(reponseBody):
    responseBodyToReturn = {}

    totalDays = len(reponseBody.get('address'))

    for i in range(totalDays):
        responseBodyToReturn.update({i : []})

    for i in range((totalDays)):
        place = {}
        place['address'] = reponseBody.get('address')[i] 
        place['experience'] = reponseBody.get('experience')[i]
        place['image'] = reponseBody.get('image')[i]
        place['location'] = reponseBody.get('location')[i]
        place['name'] = reponseBody.get('name')[i]
        place['rating'] = reponseBody.get('rating')[i]
        responseBodyToReturn.get(i).append(place)
    
    return responseBodyToReturn
