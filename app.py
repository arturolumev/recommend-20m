from flask import Flask, render_template, request, make_response, g
from math import sqrt
import os
import socket
import random
import json
import logging

import csv
from math import sqrt
from builtins import zip

option_a = os.getenv('OPTION_A', "Persona 1")
option_b = os.getenv('OPTION_B', "Persona 2")
hostname = socket.gethostname()

app = Flask(__name__)

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
        }

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

################################

def manhattan(rating1, rating2):
    """Computes the Manhattan distance. Both rating1 and rating2 are dictionaries
       of the form {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
    distance = 0
    commonRatings = False 
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
            commonRatings = True
    if commonRatings:
        return distance
    else:
        return -1 #Indicates no ratings in common


def computeNearestNeighbor(username, users):
    """creates a sorted list of users based on their distance to username"""
    distances = []
    for user in users:
        if user != username:
            distance = manhattan(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort()
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find nearest neighbor
    nearest = computeNearestNeighbor(username, users)[0][1]

    recommendations = []
    # now find bands neighbor rated that user didn't
    neighborRatings = users[nearest]
    userRatings = users[username]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))
    # using the fn sorted for variety - sort is more efficient
    return sorted(recommendations, key=lambda artistTuple: artistTuple[1], reverse = True)

#print( recommend('Hailey', users))
result = recommend('Hailey', users)

################################################################

################################################################
# USANDO CSV

def load_users_from_csv(filename):
    users = {}
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            userId = row['userId']
            movieId = row['movieId']
            rating = float(row['rating'])

            if userId not in users:
                users[userId] = {}
            users[userId][movieId] = rating

    return users

def pearson_correlation(rating1, rating2):
    sum_rating1 = sum(rating1.values())
    sum_rating2 = sum(rating2.values())
    sum_sq_rating1 = sum([rating ** 2 for rating in rating1.values()])
    sum_sq_rating2 = sum([rating ** 2 for rating in rating2.values()])

    n = min(len(rating1), len(rating2))
    if n == 0:
        return 0.0  # No common items, return 0 to avoid division by zero

    numerator = 0
    denominator1 = 0
    denominator2 = 0

    for item in rating1:
        if item in rating2:
            rating1_diff = rating1[item] - (sum_rating1 / n)
            rating2_diff = rating2[item] - (sum_rating2 / n)
            numerator += rating1_diff * rating2_diff
            denominator1 += rating1_diff ** 2
            denominator2 += rating2_diff ** 2

    denominator = sqrt(denominator1 * denominator2)
    if not denominator:
        return 0.0
    else:
        return numerator / denominator

def computeNearestNeighbor(username, users):
    distances = []
    for user in users:
        if user != username:
            distance = pearson_correlation(users[user], users[username])
            distances.append((distance, user))
    distances.sort(reverse=True)
    return distances

def recommend(username, users):
    nearest = computeNearestNeighbor(username, users)[1:6]

    movies_rated_by_user = set(users[username].keys())

    recommendations = {}
    for neighbor in nearest:
        neighbor_username = neighbor[1]
        neighbor_movies = users[neighbor_username]

        for movie in neighbor_movies:
            if movie not in movies_rated_by_user:
                if movie not in recommendations:
                    recommendations[movie] = neighbor_movies[movie]
                else:
                    recommendations[movie] += neighbor_movies[movie]

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    print(f"Las recomendaciones para {username} son:")
    for movie, rating in sorted_recommendations[:5]:
        print(f"Película: {movie}, Puntuación: {rating}")

# Ejemplo de uso
users2 = load_users_from_csv('ratings.csv')
username = '1'  # Replace this with the desired user ID for recommendations
recommend(username, users2)
print(pearson_correlation(users2['1'], users2['5']))

################################################################

@app.route("/", methods=['POST', 'GET'])
def distancias():
    if request.method == 'POST':
        user_1 = request.form['option_a']
        user_2 = request.form['option_b']
        distancia_manhattan = str(manhattan(users[user_1], users[user_2])) 
        distancia_pearson = str(pearson_correlation(users[user_1], users[user_2]))
        # Simplemente imprimir los valores en la consola
        print(f"Distancia Manhattan entre {user_1} y {user_2}: {distancia_manhattan}")
        print(f"Coeficiente de correlación de Pearson entre {user_1} y {user_2}: {distancia_pearson}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
