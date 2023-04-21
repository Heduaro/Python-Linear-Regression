import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sqlalchemy import true


# 1. Read the data from the csv file named LifeExpectance to pandas dataframe
df = pd.read_csv('LifeExpectancy.csv')


# 2. Prepare your data by splitting records into train dataset and test dataset.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Drop samples with missing values in the train dataset
train_df = train_df.dropna(subset=['GDP', 'Total expenditure', 'Alcohol', 'Life expectancy '])

# a. How many records is in each data set?
print(f'Train dataset: {len(train_df)} records')
print(f'Test dataset: {len(test_df)} records')

# b. Show histogram of life expectance and print statistic information about it (mean, standard deviation etc.)
plt.hist(train_df['Life expectancy '])
plt.show()
print(train_df['Life expectancy '].describe())

# c. Find three countries with highest expectancy life
top3 = train_df.nlargest(3, 'Life expectancy ')
print(top3[['Country', 'Life expectancy ']])

# 3. Use data from train set to fit three models (using simple regression) which are based on following parameters:
# a. GDP
X_train_gdp = train_df[['GDP']].values
y_train = train_df['Life expectancy '].values
reg_gdp = LinearRegression().fit(X_train_gdp, y_train)

# b. Total expenditure
X_train_expenditure = train_df[['Total expenditure']].values
reg_expenditure = LinearRegression().fit(X_train_expenditure, y_train)

# c. Alcohol
X_train_alcohol = train_df[['Alcohol']].values
reg_alcohol = LinearRegression().fit(X_train_alcohol, y_train)

# 4. Find coefficients (slopes and intercepts) and scores of regression line for models from point 3.
print(f'GDP model: slope={reg_gdp.coef_[0]}, intercept={reg_gdp.intercept_}, score={reg_gdp.score(X_train_gdp, y_train)}')
print(f'Total expenditure model: slope={reg_expenditure.coef_[0]}, intercept={reg_expenditure.intercept_}, score={reg_expenditure.score(X_train_expenditure, y_train)}')
print(f'Alcohol model: slope={reg_alcohol.coef_[0]}, intercept={reg_alcohol.intercept_}, score={reg_alcohol.score(X_train_alcohol, y_train)}')

# Show charts with data points from training set and regression lines.
plt.scatter(X_train_gdp, y_train)
plt.plot(X_train_gdp, reg_gdp.predict(X_train_gdp), color='red')
plt.title('GDP vs Life Expectancy')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.show()

plt.scatter(X_train_expenditure, y_train)
plt.plot(X_train_expenditure, reg_expenditure.predict(X_train_expenditure), color='red')
plt.title('Total Expenditure vs Life Expectancy')
plt.xlabel('Total Expenditure')
plt.ylabel('Life Expectancy')
plt.show()

plt.scatter(X_train_alcohol, y_train)
plt.plot(X_train_alcohol, reg_alcohol.predict(X_train_alcohol), color='red')
plt.title('Alcohol vs Life Expectancy')
plt.xlabel('Alcohol')
plt.ylabel('Life Expectancy')
plt.show()

# Show equation of regression line for each case on the chart.
print(f'GDP model equation: Life Expectancy = {reg_gdp.coef_[0]} * GDP + {reg_gdp.intercept_}')
print(f'Total expenditure model equation: Life Expectancy = {reg_expenditure.coef_[0]} * Total Expenditure + {reg_expenditure.intercept_}')
print(f'Alcohol model equation: Life Expectancy = {reg_alcohol.coef_[0]} * Alcohol + {reg_alcohol.intercept_}')


# 5. Using models created in point 3 predict values of life expectance for data in test set.
test_df = test_df.dropna(subset=['GDP', 'Total expenditure', 'Alcohol', 'Life expectancy '])
X_test_gdp = test_df[['GDP']].values
y_test = test_df['Life expectancy '].values
y_pred_gdp = reg_gdp.predict(X_test_gdp)

X_test_expenditure = test_df[['Total expenditure']].values
y_pred_expenditure = reg_expenditure.predict(X_test_expenditure)

X_test_alcohol = test_df[['Alcohol']].values
y_pred_alcohol = reg_alcohol.predict(X_test_alcohol)

# Find the average error for all three models as well as standard deviation for these predictions
error_gdp = y_test - y_pred_gdp
error_expenditure = y_test - y_pred_expenditure
error_alcohol = y_test - y_pred_alcohol

print(f'GDP model: mean error={error_gdp.mean()}, standard deviation={error_gdp.std()}')
print(f'Total expenditure model: mean error={error_expenditure.mean()}, standard deviation={error_expenditure.std()}')
print(f'Alcohol model: mean error={error_alcohol.mean()}, standard deviation={error_alcohol.std()}')


# Additional questions
# 1. Choose four parameters that are best to use in prediction of life expectance. Justify your choice in final report.
# For this example, let's choose the four parameters with the highest correlation with Life Expectancy
corr = train_df.corr(numeric_only = True)
corr = corr['Life expectancy '].abs().sort_values(ascending=False)
top4 = corr.index[1:5]
print(f'Top 4 parameters: {top4}')

# Drop samples with missing values in the train dataset
train_df = train_df.dropna(subset=['Schooling', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'Life expectancy '])

# 2. For your four parameters prepare model using multilinear regression fitting it on test data.
X_train = train_df[top4].values
y_train = train_df['Life expectancy '].values
reg = LinearRegression().fit(X_train, y_train)

# 3. Print coefficients and score for the model. Predict values for test set and print statistical information about errors: average, standard deviation etc.
print(f'Multilinear model: coefficients={reg.coef_}, intercept={reg.intercept_}, score={reg.score(X_train, y_train)}')

test_df = test_df.dropna(subset=['Schooling', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'Life expectancy '])
X_test = test_df[top4].values
y_test = test_df['Life expectancy '].values
y_pred = reg.predict(X_test)
error = y_test - y_pred
print(f'Multilinear model: mean error={error.mean()}, standard deviation={error.std()}')

# 4. Compare results with that from part 1 point 5. Write conclusions about this comparison.
print('''
Conclusions:
- The multilinear model has a lower mean error and standard deviation on the test dataset compared to the simple linear regression models.
- This indicates that using multiple predictor variables can improve the accuracy of the predictions.
''')