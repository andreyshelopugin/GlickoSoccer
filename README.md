## Glicko-2 model adapted for soccer.

Based on the paper by Andrei Shelopugin and Alexander Sirotkin titled '[Ratings of European and South American Football Leagues
Based on Glicko-2 with Modifications](https://arxiv.org/abs/2310.11459)'.

## Abstract
This model endeavors to compute club ratings of the first and second European and South American leagues. In order to
calculate these ratings, the authors have designed the Glicko-2 rating system based approach, which overcomes some Glicko-2
limitations. Particularly, the authors took into consideration the probability of the draw, the home-field advantage, and the
property of teams to become stronger or weaker following their league transitions. Furthermore, authors have constructed a
predictive model for forecasting match results based on the number of goals scored in previous matches. The metrics utilized in
the analysis reveal that the Glicko-2 based approach exhibits a marginally superior level of accuracy when compared to the
commonly used Poisson regression based approach. In addition, Glicko-2 based ratings offer greater interpretability and can find
application in various analytics tasks, such as predicting soccer players metrics for forthcoming seasons or the detailed analysis
of a players performance in preceding matches.

## Introduction
You can find useful examples in jupyter notebooks:

QuickStart.ipynb: A gentle introduction to the Glicko-2 rating system.

Boostings.ipynb: Examples of how to train and use LightGBM and CatBoost models for predicting soccer match outcomes.

Ratings.ipynb: How to use our approach to calculate club and league ratings.

## Citation

If you find this model helpful, please cite our work.

```
@misc{shelopugin2023ratings,
      title={Ratings of European and South American Football Leagues Based on Glicko-2 with Modifications}, 
      author={Andrei Shelopugin and Alexander Sirotkin},
      year={2023},
      eprint={2310.11459},
      archivePrefix={arXiv},
      primaryClass={stat.AP}
}
```