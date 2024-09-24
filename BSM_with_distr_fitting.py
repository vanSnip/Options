import yfinance as yf
import datetime
from datetime import timedelta
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Input
asset = "JPM"
R = 12 * 2  # range of the strike prices
r = 0.04  # risk free rate
Target_date = [2024, 10, 18]


# Code
print("checkpoint alpha") ######


# Importing data
def get_data(symbol, days, interval='1d'):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    data = yf.download(symbol, start=start_date,
                       end=end_date, interval=interval)
    return data


def returns(dataframe, periods=1):
    dataframe['returns'] = dataframe['Adj Close'].pct_change(periods=periods)
    return dataframe


def return_options(prices, Ks, range_begin, range_end, option_types, stock_price=0, plot=True):
    x_axes = np.arange(range_begin, range_end, 0.01)
    # Initialize net return array (absolute values)
    net_return = np.zeros_like(x_axes)

    for price, K, option_type in zip(prices, Ks, option_types):
        y_axes = []
        for i in x_axes:
            if option_type == 'call':
                if price >= 0:  # Buying call
                    if K <= i:
                        # Calculate absolute return
                        y_axes.append((i - K) - price)
                    else:
                        y_axes.append(-price)  # Calculate absolute return
                else:  # Selling call
                    if K <= i:
                        # Calculate absolute return
                        y_axes.append((K - i) - price)
                    else:
                        y_axes.append(-price)  # Calculate absolute return
            else:  # Put option
                if price >= 0:  # Buying put
                    if K >= i:
                        # Calculate absolute return
                        y_axes.append((K - i) - price)
                    else:
                        y_axes.append(-price)  # Calculate absolute return
                else:  # Selling put
                    if K >= i:
                        # Calculate absolute return
                        y_axes.append((i - K) - price)
                    else:
                        y_axes.append(-price)  # Calculate absolute return

        net_return += np.array(y_axes)

        if stock_price != 0:
            stock_ret = x_axes - stock_price
            net_return += stock_ret  # Aggregate net return (absolute values)

    if plot:
        # Plot the net return in dollars
        plt.figure(figsize=(6, 6))
        plt.plot(x_axes, net_return, color='blue', label='Net Return')
        plt.xlabel('Underlying Price')
        plt.ylabel('Net Option Return')
        plt.title('Net Return of Options')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        return (pd.DataFrame({'X-axes': x_axes, 'Net Return': net_return}))


# Going for average return in 30 days, so 30 days untill expiration
data = get_data(asset, 2560)
data = returns(data, 30)
data

returns_data = data['returns'].dropna()

bins = 100


print("checkpoint bravo") #####


# Create a histogram of the returns
plt.figure(figsize=(8, 6))
plt.hist(returns_data, bins=bins, edgecolor='black')
plt.title(f'Histogram of Returns over a Period of a Month of {asset} ')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print(returns_data)

# Fitting
# matplotlib inline

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')



# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format(ii+1, len(_distn_names), distribution))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))

        except Exception:
            pass

    return sorted(best_distributions, key=lambda x: x[2])


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc,
                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc,
                   scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


print("checkpoint charlie")##########


###
ax = returns_data.plot(kind='hist', bins=50, density=True, label="Data",
                       alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# Save plot limits
dataYLim = ax.get_ylim()
###

# find best fit
# Find best fit distribution
best_distibutions = best_fit_distribution(returns_data, 200, ax)
best_dist = best_distibutions[0]
###

# Update plots
ax.set_ylim(dataYLim)

# Make PDF with best params
pdf = make_pdf(best_dist[0], best_dist[1])

# Display the fit
plt.figure(figsize=(12, 8))
ax = pdf.plot(lw=2, label=best_dist[0].name, legend=True)
returns_data.plot(kind='hist', bins=50, density=True,
                  alpha=0.5, label="Data", legend=True, ax=ax)
####

###
param_names = (best_dist[0].shapes + ', loc, scale').split(
    ', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v)
                      for k, v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)

ax.set_title(dist_str)

S = data["Adj Close"].iloc[-1]


print("checkpoint Delta")##########


# generate a Strike price list
round_stock_price = round(S)
K_minus = [0] *  R 
K_plus = [0] *  R 
for i in range(0,  R):
    K_minus[i] = max(round_stock_price - (R/2)  + i * (0.5), 0)

for i in range(0,  R):
    K_plus[i] = round_stock_price + i *  (0.5)

K = K_minus + K_plus

print(round_stock_price)

distribution = eval(f"stats.{dist_str}")


print("Checkpoint Echo")############


# Option calc

# Days to expiry calc
# Define the function to calculate days until the target date
def days_calc(year, month, day):
    # Get today's date
    today = datetime.today()
    
    # Define the target date
    target_date = datetime(year, month, day)
    
    # Calculate the difference in days
    delta = target_date - today
    
    # Number of days
    T_days = delta.days
    return T_days

T_days = days_calc(*Target_date)

# print(f"Days until target date: {T_days}")  # target date y/m/d


print("Checkpoint Foxtrot") ###########


# Body of the black scholes calculation
# Body of the black scholes calculation
def black_scholes(name, K, S, T_days, r, option_type='call', show_volatility=False, show_price=False, over_write_price=0, over_write_sigma=0):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    data_option = yf.download(name, start_date, end_date)
    
    if over_write_price > 0:
        S = over_write_price

    if show_price:
        plt.figure(figsize=(10, 6))
        plt.plot(data["Close"], label='Price', color='black')
        plt.title(f'{asset} prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("S is:", S)

    log_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    data_option.index = pd.to_datetime(data_option.index)
    weekly_volatility = log_returns.resample('W').std() * np.sqrt(252)  # weekly volatility
    monthly_volatility = log_returns.resample('M').std() * np.sqrt(252)  # Monthly volatility
    sigma = monthly_volatility.iloc[-1]  # the sd for the monthly volatility

    if show_volatility:
        plt.figure(figsize=(14, 7))

        plt.subplot(2, 1, 1)
        plt.plot(monthly_volatility, label='Monthly Price Volatility (Annualized)', color='blue')
        plt.title('Monthly Price Volatility')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(weekly_volatility, label='Weekly Price Volatility (Annualized)', color='green')
        plt.title('Weekly Price Volatility')
        plt.legend()

        plt.tight_layout()
        plt.show()
        print("Last monthly running volatility:", monthly_volatility.iloc[-1])
        print("Last weekly running volatility:", weekly_volatility.iloc[-1])
        print("Sigma is:", sigma)

    if over_write_sigma > 0:
        sigma = over_write_sigma
    T = T_days / 365  # Convert time to expiration from days to years

    ### Price calc
    def option_price(K):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * distribution.cdf(d1) - K * np.exp(-r * T) * distribution.cdf(d2)
        elif option_type == 'put':
            return K * np.exp(-r * T) * distribution.cdf(-d2) - S * distribution.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    if show_volatility == False and show_price == False:
        return [option_price(k) for k in K]

###


print("checkpoint Golf") ########


call_prices = black_scholes(asset, K, S, T_days, r, option_type="call")
put_prices = black_scholes(asset, K, S, T_days, r, option_type="put")

df = pd.DataFrame({
    'Strike Price': K,
    'Option Call Price': round(pd.Series(call_prices), 2),
    'Option Put Price': round(pd.Series(put_prices), 2)
})
print(f"\n Current price of {asset} with target date {Target_date} is {S}.\
      \n the corresponding option prices should be: \n")
print(df)
    
