# NIAI_Practical_Project
Nature-Inspired Artificial Intelligence Practical Project. The primary objective of this project is to design and implement an Evolutionary Algorithm (EA) capable of evolving autonomous controllers for Super Mario.

# Como correr o random search
dar docker compose up num terminal na pasta code
depois ir ao local host e ver em que port está

abrir anaconda prompt e meter 
conda activate NIAI
depois
cd /d "<path_to_code>"

depois meter 
python mario_random_search_gp.py 42
ou
python mario_random_search_mlp.py 42
sendo 42 uma seed (aleatório)