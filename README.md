# Muliple-Linear-Regression-Implementation
Fun project with my implementation of multiple linear regression from scratch. Visualized with plots.

![Example](https://github.com/kraslav4ik/Muliple-Linear-Regression-Implementation/blob/main/plots/Figure_Regression.png)

In this program, I realized the simpliest linear regression model with ```.fit``` and ```.predict``` methods (both cases, with and without intercept are concidered). Also, program counts **RMSE** and **R2** errors, compare results with the Scikit-learn Linear regression class, prints difference and visualize this comparison using scatters from MatPlotLib to show a dependency between real and predicted(both ways) values. Working with ```.csv``` files

Works with python 3.8

```bash
pip install -r requirements.txt
```

## Launch

```bash
cd "./Linear Regression from Scratch"
python ./regression.py
```

Program works with file "data.csv". Sample dataset is already in repo. You can work with any numeric dataset. The only thing that program requires, column with predictable variable should be the last.

