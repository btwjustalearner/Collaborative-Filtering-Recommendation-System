# Collaborative-Filtering-Recommendation-System

In this task, I build collaborative filtering recommendation systems with train reviews and use the models to predict the ratings for a pair of user and business. You are required to implement 2 cases:

• Case 1: Item-based CF recommendation system (2pts)
In Case 1, during the training process, you will build a model by computing the Pearson correlation for the business pairs that have at least three co-rated users. During the predicting process, you will use the model to predict the rating for a given pair of user and business. You must use at most N business neighbors that are most similar to the target business for prediction (you can try various N, e.g., 3 or 5).

• Case 2: User-based CF recommendation system with Min-Hash LSH (2pts)
In Case 2, during the training process, since the number of potential user pairs might be too large to compute, you should combine the Min-Hash and LSH algorithms in your user-based CF recommendation system. You need to (1) identify user pairs who are similar using their co-rated businesses without considering their rating scores (similar to Task 1). This process reduces the number of user pairs you need to compare for the final Pearson correlation score. (2) compute the Pearson correlation for the user pair candidates that have Jaccard similarity >= 0.01 and at least three co-rated businesses. The predicting process is similar to Case 1.
