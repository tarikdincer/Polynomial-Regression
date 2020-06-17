This is a python script for gradient descent polynomial regression implementation in python without using any libraries other than Pandas.

Usage:
python polynomial_regression.py dataset.csv model.txt

Example Dataset.csv Format:

,F1,F2,F3,F4,target 
1,0.8377157496342837,0.6384139212761822,0.981834414945094,0.5957754304255635,0.057391640412137845 
2,0.4027562538227457,0.05485865543468105,0.8918170342139552,0.9483892393317622,0.20508721034647115 
3,0.653701528465938,0.6793248141356534,0.7950565588100541,0.3163774972481559,0.483799822699012

Example Model.txt Format:

a_0&#42;F1^3 + a_1&#42;F_4 + a_p
