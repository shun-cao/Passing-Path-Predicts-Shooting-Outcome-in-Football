import pandas as pd
from pylab import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math
from ast import literal_eval


#######################################################################################################################
# generation of independent features
#######################################################################################################################
def overall_path_length(location_list):
    l = location_list[:-1]
    length = 0
    for i in range(len(l)):  # the distance from the shooting zone to the attacking target is not included
        length += sqrt((l[i][0][0]-l[i][1][0])**2 + (l[i][0][1]-l[i][1][1])**2)
    for i in range(len(l)):
        if i+1<len(l)-1:
            length += sqrt((l[i][1][0]-l[i+1][0][0])**2 + (l[i][1][1]-l[i+1][0][1])**2)
    length += sqrt((l[-1][1][0]-location_list[-1][0][0])**2 + (l[-1][1][1]-location_list[-1][0][1])**2)
    return length

def ratio_of_passing_length(location_list, event_list):
    length = overall_path_length(location_list)
    passLenght = 0
    for i in range(len(event_list[:-1])):
        if event_list[i] in ['Corner', 'Free Kick', 'Free kick cross', 'Free kick shot', 'Goal kick', 'Penalty', 'Throw in', 'Cross', 'Hand pass', 'Head pass', 'High pass', 'Launch', 'Simple pass', 'Smart pass', 'Shot']:
            passLenght += sqrt((location_list[i][0][0]-location_list[i][1][0])**2 + (location_list[i][0][1]-location_list[i][1][1])**2)
    if length != 0:
        return passLenght/length
    else:
        return 100000000

def start_distance(location_list):
    return sqrt((location_list[0][0][0]-105)**2 + (location_list[0][0][1]-34)**2)

def shot_distance(location_list):
    return sqrt((location_list[-1][0][0]-105)**2 + (location_list[-1][0][1]-34.)**2)

def average_distance(location_list):
    tem_l = [location_list[0][0]]
    for l in location_list[:-1]:
        if l[0] != tem_l[-1]:
            tem_l.append(l[0])
        if l[1] != tem_l[-1]:
            tem_l.append(l[1])
    if location_list[-1][0] != tem_l[-1]:
        tem_l.append(location_list[-1][0])
    dist = []
    for l in tem_l:
        dist.append(sqrt((l[0]-105)**2 + (l[1]-34)**2))
    return mean(dist)

def x_range(location_list):
    x_list_1 = [l[0][0] for l in location_list[:-1]]
    x_list_2 = [l[1][0] for l in location_list[:-1]]
    x_list = x_list_1+x_list_2
    x_list.append(location_list[-1][0][0])
    return max(x_list)-min(x_list)

def y_range(location_list):
    y_list_1 = [l[0][1] for l in location_list[:-1]]
    y_list_2 = [l[1][1] for l in location_list[:-1]]
    y_list = y_list_1+y_list_2
    y_list.append(location_list[-1][0][1])
    return max(y_list)-min(y_list)

def moving_directness(location_list):
    l = location_list[:-1]
    if overall_path_length(location_list) != 0:
        return sqrt((l[-1][0][0]-l[0][0][0])**2 + (l[-1][0][1]-l[0][0][1])**2)/overall_path_length(location_list)
    else:
        return 1000000

def possession_time(time_list):
    return (time_list[-1] - time_list[0])

def overall_moving_speed(location_list, time_list):
    length = overall_path_length(location_list[:-1])
    time = possession_time(time_list)
    return length/time

def direct_speed(location_list, time_list):
    time = (time_list[-1] - time_list[0])
    return sqrt((location_list[-1][0][0]-location_list[0][0][0])**2 + (location_list[-1][0][1]-location_list[0][0][1])**2)/time

def pass_ratio(event_list, time_list):
    number_pass = 0
    for i in range(len(event_list)-1):
        if event_list[i] in ['Corner', 'Free Kick', 'Free kick cross', 'Free kick shot', 'Goal kick', 'Penalty', 'Throw in', 'Cross', 'Hand pass', 'Head pass', 'High pass', 'Launch', 'Simple pass', 'Smart pass', 'Shot']:
            number_pass += 1
    return number_pass/possession_time(time_list)

def number_player(player_list):
    return len(set(player_list))

def centralization_possession_time(player_list, time_list):
    playerPos = {}
    for p in player_list[:-1]:
        playerPos[p] = []
    for i in range(len(time_list[:-1])):
        playerPos[player_list[i]].append(time_list[i+1]-time_list[i])
    for k in list(playerPos.keys()):
        playerPos[k] = sum(playerPos[k])
    possession_list = list(playerPos.values())
    if len(possession_list) == 1:
        return 1.
    else:
        return sum([max(possession_list)-pT for pT in possession_list])/((len(possession_list)-1)*max(possession_list))

def centralization_passing_action(player_list, event_list):
    playerPass = {}
    for p in player_list:
        playerPass[p] = 0
    for i in range(len(event_list)):
        if event_list[i] in ['Corner', 'Free Kick', 'Free kick cross', 'Free kick shot', 'Goal kick', 'Penalty', 'Throw in', 'Cross', 'Hand pass', 'Head pass', 'High pass', 'Launch', 'Simple pass', 'Smart pass', 'Shot']:
            playerPass[player_list[i]] += 1
    pass_num_list = list(playerPass.values())
    if len(pass_num_list) == 1:
        return 1.
    else:
        return sum([max(pass_num_list)-p for p in pass_num_list])/((len(pass_num_list)-1)*max(pass_num_list))

def shot_index(location_list):
    shot_locaiton = location_list[-1][0]
    return (coef[discretization(shot_locaiton)])

def acceleration_index(location_list, time_list):
    Index = coef[discretization(location_list[-1][0])]
    time = time_list[-1]-time_list[0]
    return Index/sqrt(time)

def attack_intensity(location_list, time_list):
    tem_location = location_list[:-1]
    newL = [tem_location[0][0]]
    for l in tem_location:
        if l[0] != newL[-1]:
            newL.append(l[0])
        if l[1] != newL[-1]:
            newL.append(l[1])
    if location_list[-1][0] != newL[-1]:
        newL.append(location_list[-1][0])
    attr = sum([coef[discretization(l)] for l in newL])
    return attr/(time_list[-1]-time_list[0])


#######################################################################################################################
# assign the location of an event to the center point of the corresponded rectangle
#######################################################################################################################
def discretization(location):
    resolution_x = 105/20
    resolution_y = 68/10
    x, y = location[0], location[1]
    if (x >= 105. or x <= 0.) and (y >= 68. or y <= 0.):
        x = x
        y = y
    elif (x >= 105. or x <= 0.) and y != 68. and y != 0.:
        x = x
        y = math.trunc(y / float(resolution_y)) * resolution_y + 0.5*resolution_y
    elif (y >= 68. or y <= 0.) and x != 105. and x != 0.:
        x = math.trunc(x/float(resolution_x))*resolution_x + 0.5*resolution_x
        y = y
    else:
        x = math.trunc(x/float(resolution_x))*resolution_x + 0.5*resolution_x
        y = math.trunc(y / float(resolution_y)) * resolution_y + 0.5*resolution_y
    return (round(x, 3), round(y, 3))


#######################################################################################################################
# read data
#######################################################################################################################
data = pd.read_csv('address of the data')
data['location'] = data['location'].apply(lambda x: literal_eval(x) if "[" in x else x)
data['time'] = data['time'].apply(lambda x: literal_eval(x) if "[" in x else x)
data['player'] = data['player'].apply(lambda x: literal_eval(x) if "[" in x else x)
data['event'] = data['event'].apply(lambda x: literal_eval(x) if "[" in x else x)
shot_location=[discretization(data.iloc[i]['location'][-1][0]) for i in range(len(data))]
data['shot_location'] = shot_location
data_score = data[data['score']==1]
data_unscore = data[data['score']==0]
allData = pd.concat([data_score, data_unscore])


#######################################################################################################################
# Initial definition of discreet zones for the features of "shot index", "acceleration index", and "attack intensity"
#######################################################################################################################
coef = {}
coef[(100, 0)], coef[(100, 5.0)], coef[(100, 15.0)], coef[(100, 25.0)], coef[(100, 35.0)], coef[(100, 45.0)], coef[(100, 55.0)], coef[(100, 65.0)], coef[(100, 75.0)], coef[(100, 85.0)], coef[(100, 95.0)], coef[(100, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(97.5, 0)], coef[(97.5, 5.0)], coef[(97.5, 15.0)], coef[(97.5, 25.0)], coef[(97.5, 35.0)], coef[(97.5, 45.0)], coef[(97.5, 55.0)], coef[(97.5, 65.0)], coef[(97.5, 75.0)], coef[(97.5, 85.0)], coef[(97.5, 95.0)], coef[(97.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(92.5, 0)], coef[(92.5, 5.0)], coef[(92.5, 15.0)], coef[(92.5, 25.0)], coef[(92.5, 35.0)], coef[(92.5, 45.0)], coef[(92.5, 55.0)], coef[(92.5, 65.0)], coef[(92.5, 75.0)], coef[(92.5, 85.0)], coef[(92.5, 95.0)], coef[(92.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(87.5, 0)], coef[(87.5, 5.0)], coef[(87.5, 15.0)], coef[(87.5, 25.0)], coef[(87.5, 35.0)], coef[(87.5, 45.0)], coef[(87.5, 55.0)], coef[(87.5, 65.0)], coef[(87.5, 75.0)], coef[(87.5, 85.0)], coef[(87.5, 95.0)], coef[(87.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(82.5, 0)], coef[(82.5, 5.0)], coef[(82.5, 15.0)], coef[(82.5, 25.0)], coef[(82.5, 35.0)], coef[(82.5, 45.0)], coef[(82.5, 55.0)], coef[(82.5, 65.0)], coef[(82.5, 75.0)], coef[(82.5, 85.0)], coef[(82.5, 95.0)], coef[(82.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(77.5, 0)], coef[(77.5, 5.0)], coef[(77.5, 15.0)], coef[(77.5, 25.0)], coef[(77.5, 35.0)], coef[(77.5, 45.0)], coef[(77.5, 55.0)], coef[(77.5, 65.0)], coef[(77.5, 75.0)], coef[(77.5, 85.0)], coef[(77.5, 95.0)], coef[(77.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(72.5, 0)], coef[(72.5, 5.0)], coef[(72.5, 15.0)], coef[(72.5, 25.0)], coef[(72.5, 35.0)], coef[(72.5, 45.0)], coef[(72.5, 55.0)], coef[(72.5, 65.0)], coef[(72.5, 75.0)], coef[(72.5, 85.0)], coef[(72.5, 95.0)], coef[(72.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(67.5, 0)], coef[(67.5, 5.0)], coef[(67.5, 15.0)], coef[(67.5, 25.0)], coef[(67.5, 35.0)], coef[(67.5, 45.0)], coef[(67.5, 55.0)], coef[(67.5, 65.0)], coef[(67.5, 75.0)], coef[(67.5, 85.0)], coef[(67.5, 95.0)], coef[(67.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(62.5, 0)], coef[(62.5, 5.0)], coef[(62.5, 15.0)], coef[(62.5, 25.0)], coef[(62.5, 35.0)], coef[(62.5, 45.0)], coef[(62.5, 55.0)], coef[(62.5, 65.0)], coef[(62.5, 75.0)], coef[(62.5, 85.0)], coef[(62.5, 95.0)], coef[(62.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(57.5, 0)], coef[(57.5, 5.0)], coef[(57.5, 15.0)], coef[(57.5, 25.0)], coef[(57.5, 35.0)], coef[(57.5, 45.0)], coef[(57.5, 55.0)], coef[(57.5, 65.0)], coef[(57.5, 75.0)], coef[(57.5, 85.0)], coef[(57.5, 95.0)], coef[(57.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(52.5, 0)], coef[(52.5, 5.0)], coef[(52.5, 15.0)], coef[(52.5, 25.0)], coef[(52.5, 35.0)], coef[(52.5, 45.0)], coef[(52.5, 55.0)], coef[(52.5, 65.0)], coef[(52.5, 75.0)], coef[(52.5, 85.0)], coef[(52.5, 95.0)], coef[(52.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(47.5, 0)], coef[(47.5, 5.0)], coef[(47.5, 15.0)], coef[(47.5, 25.0)], coef[(47.5, 35.0)], coef[(47.5, 45.0)], coef[(47.5, 55.0)], coef[(47.5, 65.0)], coef[(47.5, 75.0)], coef[(47.5, 85.0)], coef[(47.5, 95.0)], coef[(47.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(42.5, 0)], coef[(42.5, 5.0)], coef[(42.5, 15.0)], coef[(42.5, 25.0)], coef[(42.5, 35.0)], coef[(42.5, 45.0)], coef[(42.5, 55.0)], coef[(42.5, 65.0)], coef[(42.5, 75.0)], coef[(42.5, 85.0)], coef[(42.5, 95.0)], coef[(42.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(37.5, 0)], coef[(37.5, 5.0)], coef[(37.5, 15.0)], coef[(37.5, 25.0)], coef[(37.5, 35.0)], coef[(37.5, 45.0)], coef[(37.5, 55.0)], coef[(37.5, 65.0)], coef[(37.5, 75.0)], coef[(37.5, 85.0)], coef[(37.5, 95.0)], coef[(37.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(32.5, 0)], coef[(32.5, 5.0)], coef[(32.5, 15.0)], coef[(32.5, 25.0)], coef[(32.5, 35.0)], coef[(32.5, 45.0)], coef[(32.5, 55.0)], coef[(32.5, 65.0)], coef[(32.5, 75.0)], coef[(32.5, 85.0)], coef[(32.5, 95.0)], coef[(32.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(27.5, 0)], coef[(27.5, 5.0)], coef[(27.5, 15.0)], coef[(27.5, 25.0)], coef[(27.5, 35.0)], coef[(27.5, 45.0)], coef[(27.5, 55.0)], coef[(27.5, 65.0)], coef[(27.5, 75.0)], coef[(27.5, 85.0)], coef[(27.5, 95.0)], coef[(27.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(22.5, 0)], coef[(22.5, 5.0)], coef[(22.5, 15.0)], coef[(22.5, 25.0)], coef[(22.5, 35.0)], coef[(22.5, 45.0)], coef[(22.5, 55.0)], coef[(22.5, 65.0)], coef[(22.5, 75.0)], coef[(22.5, 85.0)], coef[(22.5, 95.0)], coef[(22.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(17.5, 0)], coef[(17.5, 5.0)], coef[(17.5, 15.0)], coef[(17.5, 25.0)], coef[(17.5, 35.0)], coef[(17.5, 45.0)], coef[(17.5, 55.0)], coef[(17.5, 65.0)], coef[(17.5, 75.0)], coef[(17.5, 85.0)], coef[(17.5, 95.0)], coef[(17.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(12.5, 0)], coef[(12.5, 5.0)], coef[(12.5, 15.0)], coef[(12.5, 25.0)], coef[(12.5, 35.0)], coef[(12.5, 45.0)], coef[(12.5, 55.0)], coef[(12.5, 65.0)], coef[(12.5, 75.0)], coef[(12.5, 85.0)], coef[(12.5, 95.0)], coef[(12.50, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(7.5, 0)], coef[(7.5, 5.0)], coef[(7.5, 15.0)], coef[(7.5, 25.0)], coef[(7.5, 35.0)], coef[(7.5, 45.0)], coef[(7.5, 55.0)], coef[(7.5, 65.0)], coef[(7.5, 75.0)], coef[(7.5, 85.0)], coef[(7.5, 95.0)], coef[(7.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(2.5, 0)], coef[(2.5, 5.0)], coef[(2.5, 15.0)], coef[(2.5, 25.0)], coef[(2.5, 35.0)], coef[(2.5, 45.0)], coef[(2.5, 55.0)], coef[(2.5, 65.0)], coef[(2.5, 75.0)], coef[(2.5, 85.0)], coef[(2.5, 95.0)], coef[(2.5, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
coef[(0, 0)], coef[(0, 5.0)], coef[(0, 15.0)], coef[(0, 25.0)], coef[(0, 35.0)], coef[(0, 45.0)], coef[(0, 55.0)], coef[(0, 65.0)], coef[(0, 75.0)], coef[(0, 85.0)], coef[(0, 95.0)], coef[(0, 100)] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
for k in list(coef.keys()):  # change the location to fit the dimensions of (105, 68)
    coef[(round(k[0]*1.05, 3), round(k[1]*0.68, 3))] = coef.pop(k)

zones = list(coef.keys())
for i in range(len(data_score)): #data_score
    x, y = data_score.iloc[i]['location'][-1][0][0]*0.68, data_score.iloc[i]['location'][-1][0][1]*1.05
    new_l = discretization((x, y))
    for z in zones:
        if new_l == z:
            coef[z] += 1
for k in list(coef.keys()):
    coef[k] = (coef[k])/len(data_score)


#######################################################################################################################
# models with ALL variables
#######################################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

forest = RandomForestClassifier(n_estimators=300, criterion='gini', class_weight='balanced')
log = LogisticRegression(random_state=16, class_weight='balanced', solver='newton-cg', fit_intercept=True)
ann = MLPClassifier(activation='logistic', solver='adam', hidden_layer_sizes=(100,), max_iter=300, random_state=2, shuffle=True)

kf = KFold(n_splits=5, shuffle=True)

score_list = [] # accuracy
specificity = [] # precision
sensitivity = [] # recall
auc = []
importance = []
intercept_list = [] # for logistic regression model
coefficient_list = [] # for logistic regression & ann models

for iter in range(50):  # 50 runs for achieving stable results
    data = pd.concat([data_score, data_unscore.sample(frac=len(data_score)/len(data_unscore), replace=False)])
    data = data.sample(frac=1, replace=False)
    data = data.reset_index()
    pathLength = []
    ratioPassLength = []
    startDist = []
    shotDist = []
    averageDist = []
    xRange = []
    yRange = []
    movingDirect = []
    possessionTime = []
    movingSpeed = []
    directSpeed = []
    passingRatio = []
    numberPlayer = []
    centralTime = []
    centralPass = []
    shotIndex = []
    accelerationIndex = []
    attackIntensity = []
    for i in range(len(data)):
        L_list = [[(l[0][0], l[0][1]*0.68/0.75), ((l[1][0], l[1][1]*0.68/0.75))] for l in data.iloc[i]['location']]
        pathLength.append(overall_path_length(L_list))
        ratioPassLength.append(ratio_of_passing_length(L_list, data.iloc[i]['event']))
        startDist.append(start_distance(L_list))
        shotDist.append(shot_distance(L_list))
        averageDist.append(average_distance(L_list))
        xRange.append(x_range(L_list))
        yRange.append(y_range(L_list))
        movingDirect.append(moving_directness(L_list))
        possessionTime.append(possession_time(data.iloc[i]['time']))
        movingSpeed.append(overall_moving_speed(L_list, data.iloc[i]['time']))
        directSpeed.append(direct_speed(L_list, data.iloc[i]['time']))
        passingRatio.append(pass_ratio(data.iloc[i]['event'], data.iloc[i]['time']))
        numberPlayer.append(number_player(data.iloc[i]['player']))
        centralTime.append(centralization_possession_time(data.iloc[i]['player'], data.iloc[i]['time']))
        centralPass.append(centralization_passing_action(data.iloc[i]['player'], data.iloc[i]['event']))
        shotIndex.append(shot_index(L_list))
        accelerationIndex.append(acceleration_index(L_list, data.iloc[i]['time']))
        attackIntensity.append(attack_intensity(L_list, data.iloc[i]['time']))
    newData = pd.DataFrame(
        {'overall path length': pathLength,
         'ratio of passing length': ratioPassLength,
         'start dist': startDist,
         'shot dist': shotDist,
         'average dist': averageDist,
         'x range': xRange,
         'y range': yRange,
         'moving directness': movingDirect,
         'possession time': possessionTime,
         'overall moving speed': movingSpeed,
         'direct speed': directSpeed,
         'passing ratio': passingRatio,
         'number of players': numberPlayer,
         'centralization of possession T': centralTime,
         'centralization of passes': centralPass,
         'shot index': shotIndex,
         'acceleration index': accelerationIndex,
         'attack intensity': attackIntensity,
         'score': data['score']
         })
    Features = list(newData.columns)
    Features.remove('score')

    # standardize each feature
    scaler = preprocessing.MinMaxScaler()
    newData[Features] = scaler.fit_transform(newData[Features])

    i = 1
    for train_index, test_index in kf.split(newData):
        X_train = newData.iloc[train_index].loc[:, Features]
        X_test = newData.iloc[test_index][Features]
        y_train = newData.iloc[train_index].loc[:, 'score']
        y_test = newData.iloc[test_index]['score']
        #model = log.fit(X_train, y_train)
        #intercept_list.append(model.intercept_[0])
        #coefficient_list.append(model.coefs_)
        #y_pred = log.predict(X_test)
        model = forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        importance.append(forest.feature_importances_)
        #model = ann.fit(X_train, y_train)
        #y_pred = ann.predict(X_test)
        #coefficient_list.append(model.coef_)
        score = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity.append(tp/(tp+fp))
        sensitivity.append(tp/(tp+fn))
        auc.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        score_list.append(score)
        i += 1


# get feature importance based on ANN results of 5-fold cross-validation and 50 repetitions
def ann_feature_importance(C, feature):
    C = np.transpose(C)
    Q = {}
    for i in range(len(feature)):
        Q[feature[i]]=[]
        for j in range(len(C)):
            tem = [abs(c) for c in C[j]]
            Q[feature[i]].append(tem[i]/sum(tem))
    for k in list(Q.keys()):
        Q[k] = sum(Q[k])
    tem_importance = list(Q.values())
    return [t*100/sum(tem_importance) for t in tem_importance]

ann_importance = [ann_feature_importance(c[0], Features) for c in coefficient_list]
ann_importance = [list(i) for i in zip(*ann_importance)]
ann_importance = [mean(im) for im in ann_importance]


# get feature importance based on ANN results of 5-fold cross-validation and 50 repetitions
forest_importance = [mean(im) for im in np.transpose(importance)]




