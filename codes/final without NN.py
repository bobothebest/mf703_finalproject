#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:20:43 2023

@author: apple
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cvxpy as cp
import random
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

# Function to calculate daily returns from the given data
def calculate_daily_returns(data):
    # Select numeric columns for calculation
    numeric_cols = data.select_dtypes(include=[np.number])
    # Calculate percentage change and fill any NaNs with 0
    daily_returns = numeric_cols.pct_change().fillna(0)
    # Keep the original index of data
    daily_returns.index = data.index
    return daily_returns

# Function to handle zero and infinity values in a DataFrame
def handle_zero_infinity_values(df):
    for column in df.columns:
        min_nonzero = df[df[column] != 0][column].min()  # Find smallest non-zero value
        df[column] = df[column].replace(0, min_nonzero)  # Replace zeros with smallest non-zero
        # Replace infinities with next value minus a small number
        df[column] = np.where(df[column] == np.inf, df[column].shift(-1) - 1e-10, df[column])
        # Handle the last row if it's infinity
        if df[column].iloc[-1] == np.inf:
            df[column].iloc[-1] = df[column].iloc[-2] - 1e-10
    return df

# Function to load data from specified file paths
def load_data():
    # File paths
    csi300_file_path = '/Users/apple/Downloads/科科/MF703_FinalProject/data/CSI300.csv'
    stock_full_file_path = '/Users/apple/Downloads/科科/MF703_FinalProject/data/stock_full_.xlsx'
    output_label_matrix_file_path = '/Users/apple/Downloads/科科/MF703_FinalProject/data/output_label_matrix_.xlsx'

    # Load data
    csi300_data = pd.read_csv(csi300_file_path, parse_dates=['Date'], index_col='Date')
    stock_full_data = pd.read_excel(stock_full_file_path, parse_dates=['Date'], index_col='Date')
    output_label_matrix_data = pd.read_excel(output_label_matrix_file_path, parse_dates=['Date'], index_col='Date')
    # Calculate daily returns and handle zero and infinity values
    stocks_returns = calculate_daily_returns(stock_full_data)
    csi300_returns = calculate_daily_returns(csi300_data)
    stocks_returns = handle_zero_infinity_values(stocks_returns)
    csi300_returns = handle_zero_infinity_values(csi300_returns)
    return csi300_returns, stocks_returns, output_label_matrix_data

# Function to generate training and testing time periods
def generate_time_periods(start_date='2018-12-17', end_date='2023-06-12', train_length=12, test_length=6):
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    train_time, test_time = [], []
    while current_date + relativedelta(months=train_length) <= end_date:
        train_start = current_date
        train_end = train_start + relativedelta(months=train_length)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_length)
        if test_end > end_date:
            test_end = end_date
        train_time.append([train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")])
        test_time.append([test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")])
        current_date += relativedelta(months=test_length)
    return train_time, test_time

# Function to divide dataset into training and backtesting sets
def train_test_data_selected():
    csi300_returns, stocks_returns, output_label_matrix_data = load_data()
    train_periods, test_periods = generate_time_periods()
    for i in range(len(train_periods)):
        train_start, train_end = train_periods[i]
        test_start, test_end = test_periods[i]
        test_end_date = datetime.strptime(test_end, '%Y-%m-%d')
        adjusted_test_end_date = (test_end_date - timedelta(days=1)).strftime('%Y-%m-%d')
        if test_start not in output_label_matrix_data.index:
            available_dates = output_label_matrix_data.index[output_label_matrix_data.index >= test_start]
            if available_dates.empty:
                print(f"No available dates after {test_start}. Skipping this period.")
                continue
            else:
                test_start = available_dates[0]
                print(f"Adjusted test start to nearest available date: {test_start}")
        selected_stocks_row = output_label_matrix_data.loc[test_start]
        included_stocks = selected_stocks_row[selected_stocks_row == 1].index.tolist()
        training_returns = stocks_returns.loc[train_start:train_end, included_stocks]
        test_returns = stocks_returns.loc[test_start:test_end, included_stocks]
        training_csi300_returns = csi300_returns.loc[train_start:train_end]
        test_csi300_returns = csi300_returns.loc[test_start:test_end]
        yield training_returns, test_returns, training_csi300_returns, test_csi300_returns, test_start, adjusted_test_end_date


# Function to calculate the tracking error between portfolio and benchmark returns
def calculate_tracking_error(portfolio_returns, benchmark_returns):
    difference = portfolio_returns - benchmark_returns
    return np.sqrt(np.sum(difference ** 2)/(len(difference)-1))

# Function to calculate cumulative return from daily returns and weights
def calculate_cumulative_return(daily_returns, weights):
    portfolio_returns = daily_returns @ weights
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return pd.Series(cumulative_returns, index=daily_returns.index)

# Function to calculate annualized return
def calculate_annualized_return(cumulative_returns, total_days):
    if total_days == 0:
        print("Error: Total days is zero. Cannot calculate annualized return.")
        return None
    return (cumulative_returns.iloc[-1] + 1) ** (252 / total_days) - 1

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.0):
    excess_returns = portfolio_returns - risk_free_rate
    annualized_return = excess_returns.mean() * 252
    annualized_volatility = excess_returns.std() * np.sqrt(252)
    return annualized_return / annualized_volatility

# Function to calculate maximum drawdown
def calculate_max_drawdown(cumulative_returns):
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - high_water_mark) / high_water_mark
    return -drawdown.min()

# Function to calculate annualized volatility
def calculate_annualized_volatility(portfolio_returns):
    return portfolio_returns.std() * np.sqrt(252)

# Function to calculate market beta
def calculate_market_beta(portfolio_returns, market_returns):
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

# Function to calculate Sortino ratio
def calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.0, target_return=0.0):
    downside_returns = np.where(portfolio_returns < target_return, portfolio_returns - risk_free_rate, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    annualized_return = (portfolio_returns.mean() - risk_free_rate) * 252
    return annualized_return / downside_deviation

# Function to perform backtesting
def backtest(test_returns, test_csi300_returns, selected_stocks_weights, selected_stocks_codes):
    filtered_test_returns = test_returns[selected_stocks_codes]
    portfolio_returns = (filtered_test_returns * selected_stocks_weights).sum(axis=1)
    tracking_error = calculate_tracking_error(portfolio_returns, test_csi300_returns.squeeze())
    cumulative_returns = calculate_cumulative_return(filtered_test_returns, selected_stocks_weights)
    cumulative_returns = pd.Series(cumulative_returns)
    cumulative_returns.index = pd.to_datetime(cumulative_returns.index)
    total_days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    annualized_return = calculate_annualized_return(cumulative_returns, total_days)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    annualized_volatility = calculate_annualized_volatility(portfolio_returns)
    market_beta = calculate_market_beta(portfolio_returns, test_csi300_returns.squeeze())
    sortino_ratio = calculate_sortino_ratio(portfolio_returns)
    return {
        "tracking_error": tracking_error,
        "cumulative_return": cumulative_returns.iloc[-1],
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "market_beta": market_beta,
        "sortino_ratio": sortino_ratio
    }

def SLSOP(daily_returns, csi300_daily_returns, included_stocks):
    # Define the function to calculate tracking error
    def tracking_error(weights):
        # Matrix multiplication of daily returns and weights to get portfolio returns
        portfolio_returns = daily_returns @ weights  
        # Align CSI300 returns with the portfolio returns for accurate comparison
        csi300_aligned_returns = csi300_daily_returns.reindex(portfolio_returns.index)
        # Calculate the difference between portfolio and benchmark returns
        difference = portfolio_returns - csi300_aligned_returns.squeeze()  
        # Return the tracking error
        return np.sqrt(np.sum(difference ** 2)/(len(difference)-1))

    # Constraint to ensure the sum of weights equals 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds to ensure weights are between 0 and 1 for each stock
    bounds = tuple((0, 1) for _ in range(len(included_stocks)))
    # Initialize weights equally across all stocks
    initial_guess = np.full(len(included_stocks), 1 / len(included_stocks))

    # Minimize the tracking error using SLSQP optimization
    opt_result = minimize(tracking_error, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimized weights
    optimized_weights = opt_result.x
    # Select the top 30 stocks based on optimized weights
    top_30_stocks = np.argsort(optimized_weights)[-30:]
    # Get weights and codes of the selected top 30 stocks
    selected_stocks_weights = optimized_weights[top_30_stocks]
    if not isinstance(included_stocks, list):
        included_stocks = included_stocks.tolist()
    selected_stocks_codes = [included_stocks[i] for i in top_30_stocks]
    #selected_stocks_codes = included_stocks[top_30_stocks]

    # Return the weights and codes of the selected stocks
    return selected_stocks_weights, selected_stocks_codes


def solve_qp(returns_data, benchmark_returns, num_assets):
    # Define the objective function for quadratic programming
    def objective(weights):
        # Calculate portfolio returns using the provided weights
        portfolio_returns = np.dot(returns_data, weights)
        # Objective function to minimize: the sum of squared deviations from benchmark returns
        return np.sum((portfolio_returns - benchmark_returns) ** 2)

    # Define the constraint that the sum of weights should equal 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # Define bounds for each weight (between 0 and 1)
    bounds = [(0, 1) for _ in range(num_assets)]
    # Set an initial guess for the weights (evenly distributed)
    init_guess = [1 / num_assets] * num_assets
    # Optimize the weights to minimize the objective function
    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
    # Return the optimized weights
    return result.x


def run_genetic_algorithm(daily_returns, csi300_daily_returns, NGEN=1, POP_SIZE=15, CARDINALITY=30, P_CROSSOVER=0.8, P_MUTATION=0.1):
    # Convert the columns (stock symbols) from daily returns to a list
    stock_pool = list(daily_returns.columns)
    # Number of stocks in the pool
    N = len(stock_pool)

    # Define genetic algorithm functions

    def generate_individual():
        # Generate a random individual (a list of stock indices)
        return random.sample(range(N), CARDINALITY)

    def initialize():
        # Initialize the population with randomly generated individuals
        return [generate_individual() for _ in range(POP_SIZE)]

    def evaluate(individual):
        # Evaluate the fitness of an individual
        # Convert stock indices to stock symbols
        selected_stocks = [stock_pool[i] for i in individual]
        num_assets = len(selected_stocks)
        # Optimize portfolio weights using quadratic programming
        weights = solve_qp(daily_returns[selected_stocks].values, csi300_daily_returns.values[:, 0], num_assets)
        portfolio_returns = np.dot(daily_returns[selected_stocks].values, weights)

        # Calculate tracking error
        tracking_error = np.sum((portfolio_returns - csi300_daily_returns.values[:, 0])**2)
        return tracking_error

    def selection(population):
        # Select one individual from the population using tournament selection
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        return parent1 if evaluate(parent1) < evaluate(parent2) else parent2

    def crossover(parent1, parent2):
        # Perform crossover between two parents to produce an offspring
        point1, point2 = sorted(random.sample(range(CARDINALITY), 2))
        offspring = list(dict.fromkeys(parent1[:point1] + parent2[point1:point2] + parent1[point2:]))
        # Ensure the offspring has the correct length
        while len(offspring) < CARDINALITY:
            offspring.append(random.choice(range(N)))
        return offspring

    def mutation(individual):
        # Mutate an individual by swapping two elements
        point1, point2 = random.sample(range(CARDINALITY), 2)
        individual[point1], individual[point2] = individual[point2], individual[point1]
        return individual

    # Genetic algorithm execution

    population = initialize()
    population.sort(key=evaluate)

    for generation in range(NGEN):
        new_population = []
        for i in range(POP_SIZE):
            # Generate offspring through crossover and mutation
            offspring = crossover(selection(population), selection(population)) if random.random() < P_CROSSOVER else selection(population)
            offspring = mutation(offspring) if random.random() < P_MUTATION else offspring
            new_population.append(offspring)
        # Sort the new population by fitness
        population = sorted(new_population, key=evaluate)

    # Get the best individual from the final population
    best = population[0]
    best_stocks = [stock_pool[i] for i in best]
    new_num_assets = len(best_stocks)
    # Optimize weights for the best portfolio
    best_weights = solve_qp(daily_returns[best_stocks], csi300_daily_returns.values[:, 0], new_num_assets)
    return best_stocks, best_weights


def time_weighted_svm(init_weights, returns, R_market, T, C=20, epsilon=0.001):
    # Validate input data types and dimensions
    if not isinstance(init_weights, np.ndarray) or init_weights.ndim != 1:
        raise ValueError("init_weights must be a 1D numpy array")
    if not isinstance(returns, np.ndarray) or returns.ndim != 2:
        raise ValueError("returns must be a 2D numpy array")
    if not isinstance(R_market, np.ndarray) or R_market.ndim != 1:
        raise ValueError("R_market must be a 1D numpy array")
    if returns.shape[0] != R_market.shape[0]:
        raise ValueError("The number of rows in returns must match the length of R_market")

    # Define variables and parameters for the optimization problem
    w = cp.Variable(returns.shape[1])  # portfolio weights
    # Lambda function for dynamically adjusting the regularization term
    alpha = 1
    lam_ = [2 / (1 + np.exp(alpha - 2 * alpha * t / T)) for t in range(1, T+1)]
    # Loss function with epsilon-insensitive loss
    L_epsilon = cp.maximum(cp.abs(returns @ w - R_market) - epsilon, 0)
    # Objective function: minimize a combination of squared weights and loss
    objective = cp.Minimize(1/2 * cp.sum_squares(w) + C * cp.sum(cp.multiply(lam_, L_epsilon)))
    # Define constraints
    z = np.ones(returns.shape[1])
    delta_i = np.array([0.50] * returns.shape[1])  # maximum weight for any asset
    min_w = np.array([0.01] * returns.shape[1])  # minimum weight for any asset
    constraints = [cp.sum(w) == 1,  # sum of weights equals 1
                   w <= delta_i * z,  # max weight constraint
                   w >= min_w * z]   # min weight constraint
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check the status of the optimization and return results
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return w.value
    else:
        raise ValueError(f"The optimization problem could not be solved: {problem.status}")

def lasso_optimization(daily_returns, csi300_daily_returns, included_stocks):
    # Filter the daily returns dataframe to include only the specified stocks
    daily_returns_included = daily_returns[included_stocks]

    # Perform Lasso regression: LassoCV automatically finds the best alpha (regularization strength) from the provided range
    # Alphas are set in a logarithmic scale between 10^-6 and 10^-1
    # random_state=1 ensures reproducibility, cv=5 sets 5-fold cross-validation
    alphas = np.logspace(-6, -1, 100)
    lasso_cv = LassoCV(alphas=alphas, random_state=1, cv=5).fit(daily_returns_included, csi300_daily_returns.values.ravel())

    #lasso_cv = LassoCV(alphas=alphas, random_state=1, cv=5).fit(daily_returns_included, csi300_daily_returns)
    
    # Extract the coefficients from the Lasso model; these are used as optimized weights
    optimized_weights = lasso_cv.coef_

    # Select the top 30 stocks based on the highest absolute weights from the Lasso regression
    top_30_stocks = np.argsort(optimized_weights)[-30:]
    selected_stocks_weights = optimized_weights[top_30_stocks]
    
    if not isinstance(included_stocks, list):
        included_stocks = included_stocks.tolist()
    selected_stocks_codes = [included_stocks[i] for i in top_30_stocks]
    
    # Normalize the weights of the selected stocks so that their sum equals 1
    # This step is crucial to ensure the portfolio weights are valid (i.e., they sum up to 100%)
    selected_stocks_weights = selected_stocks_weights / np.sum(selected_stocks_weights)

    # Return the normalized weights and the corresponding stock codes
    return selected_stocks_weights, selected_stocks_codes



# Function to generate cumulative returns for a given time period
def cumulative_return_for_period(returns, start, end):
    period_returns = returns[start:end]
    cumulative_returns = (1 + period_returns).cumprod()
    return cumulative_returns


def main():
    first_period = True
    base_date = None

    # Lists to store results from each method
    ga_results = []  # Results from the genetic algorithm method
    slsop_results = []  # Results from the SLSOP method
    lasso_results = []  # Results from the Lasso optimization method
    
    
    # Lists to store cumulative returns data from each method
    ga_cumulative_returns = pd.Series(dtype=float)
    slsop_cumulative_returns = pd.Series(dtype=float)
    lasso_cumulative_returns = pd.Series(dtype=float)

    for training_returns, test_returns, training_csi300_returns, test_csi300_returns, test_start, test_end in train_test_data_selected():
        included_stocks = training_returns.columns.tolist()
        
        #print(training_csi300_returns)
        
        # Genetic Algorithm Method
        best_stocks, init_weights = run_genetic_algorithm(training_returns, training_csi300_returns)
        selected_returns = training_returns[best_stocks].values
        selected_stocks_weights_ga = time_weighted_svm(init_weights, selected_returns, training_csi300_returns.values[:, 0], len(training_returns))
        selected_stocks_codes_ga = best_stocks
        results_ga = backtest(test_returns, test_csi300_returns, selected_stocks_weights_ga, selected_stocks_codes_ga)

        # SLSOP Method
        selected_stocks_weights_slsop, selected_stocks_codes_slsop = SLSOP(training_returns, training_csi300_returns, included_stocks)
        results_slsop = backtest(test_returns, test_csi300_returns, selected_stocks_weights_slsop, selected_stocks_codes_slsop)

        # Lasso Optimization Method
        selected_stocks_weights_lasso, selected_stocks_codes_lasso = lasso_optimization(training_returns, training_csi300_returns, included_stocks)
        results_lasso = backtest(test_returns, test_csi300_returns, selected_stocks_weights_lasso, selected_stocks_codes_lasso)

        # Set base date for the first period
        if first_period:
            base_date = test_returns.index[0]
            first_period = False
            
         # Filter test_returns for selected stocks and adjust cumulative return based on the base date
        filtered_test_returns_ga = test_returns.loc[base_date:, selected_stocks_codes_ga]
        adjusted_cumulative_return_ga = calculate_cumulative_return(filtered_test_returns_ga, selected_stocks_weights_ga)

        filtered_test_returns_slsop = test_returns.loc[base_date:, selected_stocks_codes_slsop]
        adjusted_cumulative_return_slsop = calculate_cumulative_return(filtered_test_returns_slsop, selected_stocks_weights_slsop)
         
        filtered_test_returns_lasso = test_returns.loc[base_date:, selected_stocks_codes_lasso]
        adjusted_cumulative_return_lasso = calculate_cumulative_return(filtered_test_returns_lasso, selected_stocks_weights_lasso)

        # Store and print results for each method
        ga_results.append({
            "period": (test_start, test_end),
            "selected_stocks_and_weights": list(zip(selected_stocks_codes_ga, selected_stocks_weights_ga)),
            "results": results_ga,
            "adjusted_cumulative_return": adjusted_cumulative_return_ga
        })
        
        slsop_results.append({
            "period": (test_start, test_end),
            "selected_stocks_and_weights": list(zip(selected_stocks_codes_slsop, selected_stocks_weights_slsop)),
            "results": results_slsop,
            "adjusted_cumulative_return": adjusted_cumulative_return_slsop
        })

        lasso_results.append({
            "period": (test_start, test_end),
            "selected_stocks_and_weights": list(zip(selected_stocks_codes_lasso, selected_stocks_weights_lasso)),
            "results": results_lasso,
            "adjusted_cumulative_return": adjusted_cumulative_return_lasso
        })
        
        # Store cumulative returns for each method
        ga_cumulative_returns = pd.concat([ga_cumulative_returns, adjusted_cumulative_return_ga])
        slsop_cumulative_returns = pd.concat([slsop_cumulative_returns, adjusted_cumulative_return_slsop])
        lasso_cumulative_returns = pd.concat([lasso_cumulative_returns, adjusted_cumulative_return_lasso])


    # Print the results for all methods
    for ga_result, slsop_result, lasso_result in zip(ga_results, slsop_results, lasso_results):
        print("Genetic Algorithm Method Results:")
        print(ga_result)
        print("\nSLSOP Method Results:")
        print(slsop_result)
        print("\nLasso Optimization Method Results:")
        print(lasso_result)
        print("\n" + "="*50 + "\n")
    
    # Load CSI 300 data and calculate its cumulative returns
    csi300_returns, stocks_returns, output_label_matrix_data = load_data()

    csi300_returns = csi300_returns["2019-12-17":]

    # Generate the time periods
    _, test_time = generate_time_periods()

    # Initialize an empty DataFrame to hold the cumulative returns
    cumulative_csi300_returns = pd.DataFrame()

    # Loop through each training period and calculate cumulative returns
    for period in test_time:
        start_date, end_date = period
        period_cumulative_returns = cumulative_return_for_period(csi300_returns, start_date, end_date)
        
        cumulative_csi300_returns = pd.concat([cumulative_csi300_returns, period_cumulative_returns])

    # Reset the index to match the original date range if needed
    cumulative_csi300_returns.index = pd.to_datetime(cumulative_csi300_returns.index)
    

    # Plotting the cumulative returns of all methods and the CSI 300 index
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_csi300_returns, label="CSI 300 Index Cumulative Returns", color='black')
    plt.plot(ga_cumulative_returns, label="Genetic Algorithm Cumulative Returns", linestyle='--')
    plt.plot(slsop_cumulative_returns, label="SLSOP Cumulative Returns", linestyle='-.')
    plt.plot(lasso_cumulative_returns, label="Lasso Optimization Cumulative Returns", linestyle=':')
    plt.title("Comparison of Cumulative Returns: Methods vs CSI 300 Index")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

if __name__ == "__main__":
    main()






