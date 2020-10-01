# *******************************************************
# Copyright (C) 2020 AMELLAL Oussama <ouss.amellal@gmail.com>
#
# This file is part of 'Yolov3_2_classes_mask' project.
#
# 'Yolov3_2_classes_mask' project can not be copied and/or distributed without the express permission of AMELLAL Oussama
#  *******************************************************

import cv2
import numpy as np


# Charger les poids et la configuration Yolov3 utilisée lors de l'entraînement du Réseau de Neurones Yolov3.
RNYolo = cv2.dnn.readNet('yolov3_training_last_masque.weights', 'yolov3_training_masque.cfg')


# Charger les labels de classes (Dans notre cas un seul label : 'tortue') sur lesquelles notre modèle Yolov3 a été entraîné.
label = []
with open('obj_masque.names', 'r') as f:
    label = f.read().splitlines()

# Charger l'image de test
Image = 'image8'
image_Tortue = cv2.imread(Image + '.jpg')
hauteur, largeur, _ = image_Tortue.shape

blob = cv2.dnn.blobFromImage(image_Tortue, 1/255, (608, 608), (0,0,0), swapRB=True, crop=False)
RNYolo.setInput(blob) # Définit la nouvelle valeur d'entrée pour le réseau de neurones yolov3
# Une explication détaillée de la méthode blobFromImage : Annexe 1 ci-dessous


output_layers_names = RNYolo.getUnconnectedOutLayersNames()#Renvoie les noms des Layers avec des sorties non connectées (82, 94 et 106).
layersOutputs = RNYolo.forward(output_layers_names)
# La méthode 'forward' permet d'effectuer une passe en avant à travers notre réseau Yolov3.
# Les 'layersOutputs' sont le résultat de la détection. 'layersOutputs' est un tableau qui contient toutes les informations sur
# les objets détectés, leur position et la confidence de la détection. Ainsi il ne reste plus qu'à afficher le résultat à l'écran.


boite = []# Va contenir les boîtes détectées (Les rectangles englobant chaque objet détecté)
confidences = []
class_ids = []

for output in layersOutputs:
    for detection in output:
        scores = detection[5:]#Dans le cas de détection de plusieurs classes, le variable 'scores' contient la confidence de chaque classe
        class_id = np.argmax(scores)# Dans le cas de détection de plusieurs classes, la classe de plus grand score est sélectionnée
        confidence = scores[class_id]# On récupère le score du classe sélectionnée
        if confidence > 0.5: # Les boîtes de confidence inférieure à 0.5 ne seront pas traitées.
            center_x = int(detection[0]*largeur) # Position x du centre de la boîte détectée
            center_y = int(detection[1]*hauteur) # Position y du centre de la boîte détectée
            w = int(detection[2]*largeur) #Largeur de la boîte détectée
            h = int(detection[3]*hauteur) #Hauteur de la boîte détectée

            x = int(center_x - w/2) # Position x du point en haut à gauche de la boîte détectée
            y = int(center_y - h/2) # Position y du point en haut à gauche de la boîte détectée

            boite.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

#Lorsque nous effectuons la détection, il arrive que nous ayons plus de boîtes pour le même objet,
# nous devrions donc utiliser la fonction 'NMSBoxes'( Non maximum suppresion) pour supprimer les boîtes redondantes.
indexes = cv2.dnn.NMSBoxes(boite, confidences, 0.5, 0.4)
# Une explication détaillée de la méthode NMSBoxes : Annexe 2 ci-dessous
font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(len(boite), 3)) # Initialiser une liste de couleurs pour représenter chaque label de classe possible.

for i in indexes.flatten(): # Affichage des boites sélectionnées
    x, y, w, h = boite[i]
    labelmasque = str(label[class_ids[i]])
    print(labelmasque)
    if labelmasque == "masque":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    confidence = str(round(confidences[i], 2))
    #color = colors[i]
    cv2.rectangle(image_Tortue, (x,y), (x+w, y+h), color, 2)

    if labelmasque == "masque":
        cv2.putText(image_Tortue, labelmasque + " " + confidence,(x-14, y-3), font, 0.4, color, 1 )
    else:
        cv2.putText(image_Tortue, labelmasque + " " + confidence, (x - 30, y - 3), font, 0.4, color, 1)

cv2.imshow('Image', image_Tortue)
cv2.waitKey(0)
cv2.destroyAllWindows()

filename = 'ResultatDeDetection/' + Image + 'Sortie' +'.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_Tortue)
#======================================== début Annexe 1 ==================================================

#Les arguments de la méthode  :
#image_Tortue :
# L'image à pré-traiter avant de la passer au réseau de neurones Yolov3 pour faire la détection d'objets.
#1/255 :
# 1/255  est une valeur par laquelle nous multiplierons les données avant tout autre traitement. Nos images originales
# sont constituées de coefficients RVB compris entre 0 et 255, mais ces valeurs seraient trop élevées pour que nos modèles
# puissent les traitées, nous ciblons donc des valeurs comprises entre 0 et 1 à la place, en les mettant à l'échelle avec un facteur de  1./255.
#(416, 416) :
#YOLOv3 accepte trois tailles de Blob:
#320 × 320  : résultats moins précis mais plus rapide.
#416 × 416  : compromis précision et rapidité.
#608 × 608  : résultats plus précis mais plus lent.
#(0,0,0) :
#Ce sont nos valeurs moyennes de soustraction. Ils peuvent être un 3-tuple des moyens RVB ou ils peuvent être une valeur unique,
# auquel la valeur fournie est soustraite de chaque canal de l'image.
#swapRB=True
#OpenCV suppose que les images sont dans l'ordre des canaux BGR; cependant, la valeur «moyenne» suppose que nous utilisons
# l'ordre RVB. Pour résoudre cet écart, nous pouvons permuter les canaux R et B dans l'image en définissant cette valeur
# sur «True». Par défaut, OpenCV effectue ce changement de chaîne pour nous.
#crop=False
#Une valeur de True recadrera le centre de l'image en fonction de la largeur et de la hauteur d'entrée.
# Sinon, l'image entière est utilisée.
#======================================== fin Annexe 1 ==================================================

#======================================== début Annexe 2 ==================================================
# Paramètres utilisés : cv2.dnn.NMSBoxes(boite, confidences, 0.5, 0.4)
#boite : un ensemble de boites dont on veut appliquer l'NMS.
#confidences : un ensemble de confidences correspondantes.
#0.5 : un seuil utilisé pour filtrer les boites par score.
#0.4 : paramètre expliqué dans l'algorithme suivante :

#Voici le processus de sélection de la meilleure boîte englobante d'un objet détecté à l'aide de NMS après un filtrage des boites
# avec un score de confidence supérieur à 50%(0.5) :
#Étape 1: Sélectionnez la boite avec le score de confidence le plus élevé.
#Étape 2: Ensuite, comparez le chevauchement (intersection sur union*) de cette boite avec les autres boites.
#Étape 3: Supprimez les boites de délimitation avec un chevauchement (intersection sur union)> 40%(paramètre 0.4)
#Étape 4: Ensuite, passez au prochain score du boite le plus élevé.
#Étape 5: Enfin, répétez les étapes 2 à 4.

#*l'intersection sur union entre deux boites est l'aire de l'intersection de ces deux boites divisé par l'union des aires des deux boites.
#======================================== fin Annexe 2 ==================================================