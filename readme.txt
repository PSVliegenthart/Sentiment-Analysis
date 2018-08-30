This Python script enables researchers to train a Bernoulli Bayes Classifier and deploy the classifier on fresh data. To make the operations as easy as possible, arguments can be used to operate the script. The arguments can be retrieved by running the following commend: nlclassify.py -h (from Help).

Feature overview 
-The script only works with .csv files 
-The seperator for both the training and predict file can be set to ',' or ';'
-The script can handle any number of catagories. 
-You can select the training colomn and predict colomn using the header title 
-You can set the train-test ratio 
-The script ignores missing or bad data 
-The script outputs a most informative features file for every catagory 
-The script outputs a copy of the predict file with the polarity score per catagory and the most matching catagory as a label 
-The script outputs an accuracy score 
-When the classifier is trained, it will store the classification model seperately, so that it can be loaded the next time to save time 

Arguments: 

[-h] [-T TRAIN] [-TT TRAINTEXT] [-TC TRAINCAT] [-S SPLIT]
             [-P PREDICT] [-PT PREDICTTEXT] [-PC PREDICTCAT] [-O OUTPUT]
             [-TSP TRAINSEP] [-PSP PREDICTSEP] [-F FEATURE]

Example: for the provided test files the following command starts the script: 

python nbclassify.py -T tcat_brexit_train.csv -TC Label -TT tekst -P tcat_brexit_predict.csv -PT text -PSP , 

