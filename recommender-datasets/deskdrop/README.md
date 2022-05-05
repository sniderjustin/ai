# CI&T DeskDrop

[source](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop)

Deskdrop is an internal communications platform developed by CI&T, focused in companies using Google G Suite. Among other features, this platform allows companies employees to share relevant articles with their peers, and collaborate around them.

Included are 12 months of logs with 73k user interactions about 3k public articles. 

The types of events in the logs include:

* view
* like
* comment created
* follow
* bookmark



| Metrics            | Notebook                                               | Model                                   | Frameworks    |
| ------------------ | ------------------------------------------------------ | --------------------------------------- | ------------- |
| Recall@5 of 0.3426 | [recommender-system.ipynb](./recommender-system.ipynb) | Hybrid: Collaborative and Content-Based | Numpy, Pandas |
| Recall@5 of 0.3339 | [recommender-system.ipynb](./recommender-system.ipynb) | Collaborative Filtering                 | Numpy, Pandas |
| Recall@5 of 0.2417 | [recommender-system.ipynb](./recommender-system.ipynb) | Popularity Model                        | Numpy, Pandas |
| Recall@5 of 0.1628 | [recommender-system.ipynb](./recommender-system.ipynb) | Content-Based Filtering Model           | Numpy, Pandas |



## Tutorials

* [Recommender Systems in Python](https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101/notebook) by Gabriel Moreira



## Terms



**Collaborative filtering:** makes prediction (filtering) about the interest of a single user by collecting information about the interest from many users (collaborating). The underlying assumption is if person A has the same opinion as person B, then A is more likely to agree with B in general than some random person. 

**Content-Based Filtering:** This method finds reccomends conent that is similar to content already consumed a particular user. 

**Hybrid methods:** A combined model using multiple models. Research has shown a combination of models can be more effective in many applications. 

**Data munging:** Data wrangling

**User cold-start:** Hard to reccomend content for new users since we do not know much about them. Often it is a good idea to not include new user info when training models. 

**Holdout cross-validation:** A random data sample is excluded from all training, and is used for evaluation. For this data set a better approach would be to sample past data to predict future events. 

**Test set:** Data held out to use for evaluation only.

**Top-N accuracy metrics:** evaluate the accuracy of the top recommendations provided to a user when compared to the items the user has actually interacted with in the test set. 

**Top-N accuracy metrics algorithm:**

* For each user
  * For each item the user has interacted with in the test set
    * Sample 100 other items the user has never interacted. 
    * Ask the recommender model to produce a ranked list of recommended items, from a set composed of one interacted item and the 100 non-interacted ("not-relevant") items. 
    * Compute the top-N accuracy metrics for this user and interacted item from the recommendations ranked list. 
  * Aggregate the global Top-N accuracy metrics for all users

**Recall@N:** a Top-N metric that evaluates whether the interacted item is among the top N items (hit) in the ranked list of 101 recommendations for a user. 

**NDCG@N:** ???MORE INFO HERE... better because takes into account the position of the relevant item. See here: http://fastml.com/evaluating-recommender-systems/

**MAP@N:** ????MORE INFO HERE... better because takes into account the position of the relevant item. See here: http://fastml.com/evaluating-recommender-systems/









## Frameworks



Numpy

