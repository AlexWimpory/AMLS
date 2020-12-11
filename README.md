# AMLS Coursework
This project uses machine learning techniques to perform 4 classification tasks.

Directory Structure:
* A1: Gender detection: male or female
* A2: Emotion detection: smiling or not smiling
* B1: Face shape recognition: 5 types of face shapes
* B2: Eye color recognition: 5 different eye colors
* Datasets: Celeba, Celeba_test and Cartoon_set, Cartoon_set_test

Each task is self-sufficient but the code was designed to be resuable
 therefore there is a lot of duplicate code in each directory.

Packages required found in requirements.txt and can be installed using pip install -r requirements.txt.
  However I did have some issues getting dlib to work on my Windows machine, but this should not matter as
  explained in the report, this did not end up being used.