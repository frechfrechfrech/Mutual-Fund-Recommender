# Mutual-Fund-Recommender
Using historic sales and fund characteristics to recommend relevant mutual funds to financial advisors.


# Background

## Project Motivation

- Company: Strategic Insight
- Product: Local Market Share
- Purpose: Help mutual fund wholesalers to make more informed and targeted sales to financial advisors.
- Goal: Build a recommender model to suggest mutual funds that a given office is likely to buy


## Wholesaler-Driven Mutual Fund Selling Structure

![](images/oranges.png)



## Recommender Model:

### A recommender you know and love: Netflix
![](images/top_picks_for_alex.png)
    Netflix provides recommendations for movies to its users
    One of the methods it employs is collaborative filtering
    Idea: "People like you like this movie so we think you’ll like it too."
    Steps:
        1. Use ratings/preferences of similar users to predict what ratings you would give movies/tv shows
        2. Recommend the movies/tv shows with the highest predicted ratings


### Mutual Fund Recommender Components

![](images/recommender_components.png)

### Collaborative Filtering through Matrix Decomposition

This type of recommender system seeks to predict currently-unknown ratings.
- The matrix of known ratings is decomposed with the objective of minimizing error on **known** ratings. Unknown ratings are ignored.
- The original ratings matrix is then reconstructed and the resulting values for the unknown ratings are the rating predictions.

![](images/matrix_decomp.png)

## WebApp

In order to be useful to mutual fund wholesalers who are pitching products to
financial advisors, this tool needs to be available on-the-go.

To facilitate that, I have created a flask app prototype.

### Welcome Page
<img src="images/flask_app_1.png" alt="drawing" width="1000" height="600"/>

### Make Selections
Choose the firm and zip where the advisor works.
![](images/flask_app_2.png =64x92)

### Results
The recommender returns the top funds currently selling in the office as well
as the top fund recommendations based on collaborative filtering.
![](images/flask_app_3.png =64x92)
