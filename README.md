# La détection automatique du port du masque
Entraîner votre modèle à détecter deux classes avec YOLOv3, Deep learning, Opencv, Google Colab

Tutoriel détaillé pour entraîner votre modèle à détecter deux classe ( port ou non du masque) avec YOLO version 3 
(YOLO est un réseau qui utilise des algorithmes d'apprentissage profond (Deep Learning) 
pour la détection d'objets.

Environement d'exécution :              
La partie training du modèle est faite dans Google Colaboratory, qui est un service hébergé
de notebooks jupyter qui ne nécessite aucune configuration et permet d'accéder gratuitement 
à des ressources informatiques, dont des GPU.

Étape 1 : Préparation de la base de données.
La base de données est composé de 1420 images.
Le fichier Apprentissage/Train_YoloV3_masque_.ipynb contient les étapes à suivre pour créer votre propre base de données.

Lien de téléchargement  de la BD :            
https://drive.google.com/file/d/1z16Pf0XZ_pMjEyQHdrUQ1iHL6wSFM03P/view?usp=sharing       Taille :  189 Mo

Étape 2 : Apprentissage.
Détaillé dans le fichier Apprentissage/Train_YoloV3_masque_.ipynb

Étape 3 : Tester le modèle entraîné.
Détaillé dans le fichier Tester_Le_Modele/testmasque.py

Lien de téléchargement du fichier yolov3_training_last_masque.weights   :           
https://drive.google.com/file/d/1-W6eT3N0BHjmM5s8yJvhaVIVqOMP02JP/view?usp=sharing      Taille : 235 Mo


<div align="center">
<img src="https://github.com/OAMELLAL/Yolov3_2_classes_Mask_vs_NoMask/blob/master/Tester_Le_Model/ResultatDeDetection/result_merged.png" >
<p>Résultats de détection</p>
</div>
