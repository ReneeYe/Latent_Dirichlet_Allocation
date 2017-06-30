# Latent_Dirichlet_Allocation
The group project to implement LDA algorithm


# version
python 3.6

# lda model
### LDA_Gibbs_Sampling.py 

is the major implement of gibbs LDA topic model.
### Topic_Modeling.py 

is used to divide the whole corpus into training set and testing set and then using the testing set to compute the perplexity of the given topic numbers.
### util.py 

is used to do some preprocessing.
# dataset
### experiment_food_data.csv

is the selected 1500 review data from the original dataset.

### ml-20m(too large, not included in archive)

is the file for datasets of logs of movie reviews from MovieLens 

### final_movie_2000.csv

is the selected 2000 users from rating.csv in ml-20m file


# document modelling
### topic_model_plsi.ipyn

uses gensim python library to train the text data with pLSA model, to generate topic items

# topic number selecting and document classification
### Text_lda.py 

is the process of choosing topic number according to perplexity, and also,it gives the representative words of topics and the possible topics of the document(ie. Text)
### Text_classification.py 

is the implement of classification, using LR model and compute the accuray of the LDA feature and word feature.
### Summary_lda.py

is almost the same as Text_lda.py except using Summary as the corpus.
### Summary_classification.py

is the implement of classification for Summary, using SGD model, also, I've added word2vec feature in this part. And compare the accuracy of different model.


# collaborative filtering
### final_experiment.ipynb

include all codes from parameter tuning to final model evaluation

### AI_plot.

delicated plot for CF
