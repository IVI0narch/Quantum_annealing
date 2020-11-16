from dwave.system import DWaveSampler, EmbeddingComposite
import pyqubo
import pandas_datareader.data as web
import pandas as pd

sampler = EmbeddingComposite(DWaveSampler())

total_budget = 10**6
value_contribution = [1/3, 1/3, 1/3]
cov_matrix = []

#  ticker, pe, price
stocks = [['LKOH', 13.33, 4926], ['SNGSP', 14.19, 39], ['GAZP', 10.33, 185.3], ['TATN', 9.41, 513],
          ['GMKN', 17.73, 20694], ['RUAL', 21.76, 33.6], ['ALRS', 21.13, 86.1], ['PLZL', 26.42, 15960],
          ['DSKY', 17.8, 120], ['YNDX', 67.17, 4741], ['FIVE', 33.51, 2787], ['MVID', 15.56, 731],
          ['VTBR', 3.8, 0.037], ['SBER', 6.94, 248], ['PRMB', 6.04, 25800], ['MTSS', 10.67, 321]]

oil_gas = [['LKOH', 13.33, 4926], ['SNGSP', 14.19, 39], ['GAZP', 10.33, 185.3], ['TATN', 9.41, 513]]
metallurgy_mining = [['GMKN', 17.73, 20694], ['RUAL', 21.76, 33.6], ['ALRS', 21.13, 86.1], ['PLZL', 26.42, 15960]]
consumers = [['DSKY', 17.8, 120], ['YNDX', 67.17, 4741], ['FIVE', 33.51, 2787], ['MVID', 15.56, 731]]
banks = [['VTBR', 3.8, 0.037], ['SBER', 6.94, 248], ['PRMB', 6.04, 25800], ['MTSS', 10.67, 321]]

industries = [oil_gas, metallurgy_mining, consumers, banks]
prices = [0 for i in range(len(stocks))]
cov_matrix = [[] for i in range(len(stocks))]
data = []


def pe_strategy(matrix):
    sum = 0
    k = 0
    min_pe = 100
    ticker = ''
    ticker_price = 0
    for i, elem in enumerate(matrix):
        sum += elem[1]
        k += 1
        if elem[1] < min_pe:
            min_pe = elem[1]
            ticker = elem[0]
            ticker_price = elem[2]
    mean_pe = sum / k
    delta_pe = abs(mean_pe - min_pe)
    potential = abs(1 - delta_pe/min_pe) * ticker_price
    return potential, ticker, ticker_price

top_industry_potentials = []
top_industry_stocks = []
top_industry_prices = []
for elem in industries:
    potential, ticker, ticker_price = pe_strategy(elem)
    top_industry_potentials.append(potential)
    top_industry_stocks.append(ticker)
    top_industry_prices.append(ticker_price)


for i, elem in enumerate(top_industry_stocks):
    data += [web.DataReader(elem, 'moex', start='2019-01-01', end='2020-01-01')]
    prices[i] = data[i]['HIGH']

for i in range(len(top_industry_stocks)):
    for j in range(len(top_industry_stocks)):
        cov_ij = (pd.Series((prices[i] - prices[i].mean()) * (prices[j] - prices[j].mean()))).mean()
        cov_matrix[i] += [cov_ij]

x = pyqubo.Array.create('x', shape=len(top_industry_stocks), vartype='BINARY')

H1 = -sum(a * expected_return for a, expected_return in zip(x, top_industry_potentials))

for i in range(len(top_industry_stocks)):
    for j in range(len(top_industry_stocks)):
        H2 = x[i] * x[j] * cov_matrix[i][j]

H3 = (sum(a * price for a, price in zip(x, top_industry_prices)) - total_budget)**2

H = H1 + H2 + H3
Q, offset = H.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=1000)

print(' '.join(str(elem) for elem in top_industry_stocks))
print(sampleset)
