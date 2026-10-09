# FIFA World Cup Match Prediction & Tournament Simulation

A machine learning model that predicts international football match outcomes from FIFA rankings and recent team form, then simulates a full World Cup from group stage to final. It was a course assignment for *Pengantar Sains Data* with the theme of predicting the 2026 World Cup.

![World Cup simulation result](https://github.com/harrymardika/prediction-world-cup/assets/130530985/008df9e7-5d28-4466-bbca-01a10baa236e)

## Results

| Model | Training accuracy | Test accuracy |
|---|---|---|
| Gradient Boosting (GridSearchCV, 576 candidates) | 74.83% | 69.28% |
| **Random Forest** (GridSearchCV, 960 candidates) | 84.34% | **70.50%** |

The Random Forest was used for the simulation. Best parameters: `max_depth=10`, `max_features="sqrt"`, `max_leaf_nodes=100`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=150`.

### Tournament simulation

The 2026 draw was not available, so the simulation uses the 32-team groups of the 2022 World Cup with team statistics up to November 2023. Group winners: Netherlands, England, Argentina, France, Spain, Belgium, Brazil, Portugal. Semi-finals: Argentina beat Spain (probability 0.53) and England beat Portugal (0.56). In the final, **England beat Argentina** with probability 0.54.

## Data

| File | Description |
|---|---|
| `Dataset/results.csv` | 45,315 international results, 1872 to November 2023 |
| `Dataset/fifa_ranking-2023-07-20.csv` | FIFA men's ranking history |
| `Dataset/Final/model_df.csv` | Modeling dataset: 3,695 matches with engineered features |

Only matches since August 2018 are used (4,889 matches), so features describe the current World Cup cycle.

## Approach

1. **Preparation:** filter to matches since August 2018, forward-fill the ranking to daily values, harmonize country names, and join each match with both teams' ranking on the match date.
2. **Target:** 0 if the home team wins, 1 otherwise (draws merged with losses). Classes are balanced (1,776 vs. 1,919).
3. **Features:** per team, over the World Cup cycle and the last 5 matches: goals scored and conceded, average opponent rank, and match points weighted by opponent rank. The model uses differences between the two teams plus a friendly-match flag.
4. **Modeling:** 80/20 split, Gradient Boosting and Random Forest tuned with 3-fold `GridSearchCV`.
5. **Simulation:** each match is predicted twice with the teams swapped (no home advantage at a World Cup) and the higher average probability wins. If the two predictions disagree on the favorite, the group match is a draw. The top two per group advance, and the knockout bracket is drawn with NetworkX.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, NetworkX, pydot (Graphviz needed for the bracket layout).

## Project Structure

```
prediction-world-cup/
├── wc-perdiction.ipynb      # Preparation, features, models, simulation
└── Dataset/                 # results.csv, fifa_ranking-2023-07-20.csv, Final/model_df.csv
```

## Getting Started

```bash
git clone https://github.com/harrymardika/prediction-world-cup.git
cd prediction-world-cup
pip install pandas numpy scikit-learn matplotlib seaborn networkx pydot jupyter
jupyter notebook wc-perdiction.ipynb
```

## Limitations

- Draws are merged with losses in training, so the model does not predict draws directly.
- The simulation uses the 2022 groups; it needs the 2026 draw (48 teams, 12 groups) and updated rankings.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
