from flask import Flask, render_template, request, make_response, g
from math import sqrt
import os
import socket
import random
import json
import logging

import csv
from math import sqrt
import os

### 10M
import fileinput
import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock

### 20M
import math
import pandas as pd
import dask.dataframe as dd

option_a = os.getenv('OPTION_A', "Persona 1")
option_b = os.getenv('OPTION_B', "Persona 2")
hostname = socket.gethostname()

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

################################################################

# 20M

# Init K-vector with correct value based on distance type
def initVectDist(funName, N):
    if funName == 'euclidiana' or funName == 'manhattan' or funName == 'euclidianaL' or funName == 'manhattanL':
        ls = [99999] * N
    else:
        ls = [-1] * N

    lu = [None] * N
    return ls, lu


# Keep the closest values, avoiding sort
def keepClosest(funname, lstdist, lstuser, newdist, newuser, N):
    if funname == 'euclidiana' or funname == 'manhattan' or funname == 'euclidianaL' or funname == 'manhattanL':
        count = -1
        for i in lstdist:
            count += 1
            if newdist > i:
                continue
            lstdist.insert(count, newdist)
            lstuser.insert(count, newuser)
            break
    else:
        count = -1
        for i in lstdist:
            count += 1
            if newdist < i:
                continue
            lstdist.insert(count, newdist)
            lstuser.insert(count, newuser)
            break

    if len(lstdist) > N:
        lstdist.pop()
        lstuser.pop()
    return lstdist, lstuser

def readLargeFile(filename, delim=','):
  data = pd.read_csv(filename, delimiter=delim, header=None)

  lst = {} # Dictionary
  j = 0
  for index, row in data.iterrows():
      #print(row[0], row[1], row[2])
      if j != row[0]:
          j = row[0]
          tmp = {row[1]:row[2]}
          lst[row[0]] = tmp
      else:
          tmp = lst.get(row[0])
          tmp[row[1]] = row[2]
          lst[row[0]] = tmp
  return lst

def readLargeFileDask(filename, delim=','):
    # Utiliza assume_missing=True para manejar valores no especificados en la conversión a int64
    ddf = dd.read_csv(filename, delimiter=delim, header=None, assume_missing=True)

    ddf_grouped = ddf.groupby(0).apply(lambda group: dict(zip(group[1], group[2])), meta=('x', 'f8'))

    result = ddf_grouped.compute().to_dict()

    return result

# Euclidian distance
def euclidianaL(user1, user2):
    dist = 0.0
    count = 0
    for i in user2:
        if not (user1.get(i) is None):
            x = user1.get(i)
            y = user2.get(i)
            dist += math.pow(x - y, 2)
            count += 1

    if count == 0:
        return 9999.99
    return math.sqrt(dist)

# Manhattan distance
def manhattanL(user1, user2):
    dist = 0.0
    count = 0
    for i in user2:
        if not (user1.get(i) is None):
            x = user1[i]
            y = user2[i]
            dist += abs(x - y)
            count += 1

    if count == 0:
        return 9999.99
    return dist


# Cosine distance
def cosenoL(user1, user2):
    xy = xx = yy = 0.0
    count = 0
    for i in user2:
        if not (user1.get(i) is None):
            x = user1[i]
            y = user2[i]
            xy += x * y
            xx += x * x
            yy += y * y
            count += 1

    den = math.sqrt(xx) * math.sqrt(yy)
    if den == 0:
      return math.nan
    if count == 0:
        return math.nan
    return xy / den  # -1 a +1


# Pearson distance
def pearsonL(user1, user2):
    xy = zx = zy = n = zxx = zyy = 0.0
    count = 0
    for i in user2:
        if not (user1.get(i) is None):
            x = user1[i]
            y = user2[i]
            xy += x * y
            zx += x
            zy += y
            n += 1
            zxx += x * x
            zyy += y * y
            count += 1

    if n == 0:
        return math.nan
    a = xy - (zx * zy) / n
    b = math.sqrt(zxx - (zx * zx) / n) * math.sqrt(zyy - (zy * zy) / n)
    if b == 0:
        return math.nan
    if count == 0:
        return math.nan
    return a / b  # -1 a +1

# K-Nearest neighbour
def knn_L(N, distancia, usuario, data):  # N numero de vecinos
    funName = distancia.__name__
    print('k-nn', funName)

    listDist, listName = initVectDist(funName, N)
    nsize = len(data)
    otherusers = range(0, nsize)
    vectoruser = data.get(usuario)

    for i in range(0, nsize):
        tmpuser = i
        if tmpuser != usuario:
            tmpvector = data.get(tmpuser)
            if not (tmpvector is None):
              tmpdist = distancia(vectoruser, tmpvector)
              if tmpdist is not math.nan:
                listDist, listName = keepClosest(funName, listDist, listName, tmpdist, tmpuser, N)

    return listDist, listName

# Find the K closest firsts Item recommendation
def recommendationL(usuario, distancia, N, items, minr, data):
    ldistK, luserK = knn_L(N, distancia, usuario, data)

    user = data.get(usuario)
    recom = [None] * N
    for i in range(0, N):
        recom[i] = data.get(luserK[i])
    # print('user preference:', user)

    lstRecomm = [-1] * items
    lstUser = [None] * items
    lstObj = [None] * items
    k = 0

    fullObjs = {}
    count = 0
    for i in recom:
        for j in i:
          tmp = fullObjs.get(j)
          if tmp is None:
            fullObjs[j] = [i.get(j), luserK[count]]
          else:
            nval = i.get(j)
            if nval > tmp[0]:
              fullObjs[j] = [nval, luserK[count]]
        count += 1

    finallst = topSuggestions(fullObjs, count, items)
    return finallst

def topSuggestions(fullObj, k, items):
  rp = [-1]*items

  for i in fullObj:
    rating = fullObj.get(i)

    for j in range(0, items):
      if rp[j] == -1 :
        tmp = [i, rating[0], rating[1]]
        rp.insert(j, tmp)
        rp.pop()
        break
      else:
        tval = rp[j]
        if tval[1] < rating[0]:
          tmp = [i, rating[0], rating[1]]
          rp.insert(j, tmp)
          rp.pop()
          break

  return rp

lstdb20 = readLargeFileDask('ratings.csv') # 54 seg

usuario = 45600
rfunc = euclidianaL

# 10 vecinos, 20 recomendaciones
lista = recommendationL(usuario, rfunc, 10, 20, 3.0, lstdb20) # 3 seg
for i in lista:
    print('user:', i[2], 'obj:',i[0], 'rating:', i[1])

################################################################

@app.route("/", methods=['POST', 'GET'])
def distancias():
    if request.method == 'POST':
        user_1 = request.form['option_a']
        user_2 = request.form['option_b']
        recomendacion_20m = str(lista) 
        # Simplemente imprimir los valores en la consola
        print(f"Recomendacion 20 Millones: {recomendacion_20m}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)

