---
title: "Disaggregated Price Stock Screener"
subtitle: "Building a Value-Ranked Screener with Finviz and Python"
author: "Vikram Oddiraju"
date: 2024-10-26
summary: "A quantitative walkthrough of building a ranked stock screener based on disaggregated price using Finviz data and the DuPont Identity."
editPost:
    URL: "https://vikramoddiraju.substack.com/publish/post/149990109"
    Text: "Substack Version"
draft: false
---

## TL;DR

If you don’t feel like reading this newsletter but want the code, here’s the link to my GitHub repository:  
[**Disaggregated Price Stock Screener**](https://github.com/yourusername/Disaggregated-Price-Stock-Screener)

The code is well documented, but the last section of this post walks you through it step-by-step.

---

> *"A body at rest will remain at rest, and a body in motion will remain in motion in the direction of its motion unless acted upon by an outside force."*  
> — **Sir Isaac Newton (First Law of Motion)**

---

Newton probably hadn’t predicted this, but his first law of motion is just as prudent for solving a rigid-body physics problem as it is for addressing one of the hardest hurdles facing individual investors.

In pursuit of actively managing a personal portfolio, many individual investors get stuck at the **“getting the ball rolling”** step — finding what companies to perform due diligence on.  
This barrier to entry was one of the reasons **Jack Bogle** created the *Vanguard 500 Index Fund* — a passively managed mutual fund that tracks the S&P 500, giving investors diversification without excessive management fees.  
It’s also why passive funds’ AUM overtook that of actively managed funds at the end of 2023.

While there’s nothing wrong with being a passive investor (in fact, most people should be), I believe that individual investors would benefit from **some of the tools professional investors use** — one being the **stock screener**.

---

## Why Screeners Matter

The main purpose of a stock screener is to quickly filter out companies that don’t fit an investor’s criteria.  
For example, an investor might say:

> “I only want companies with ROIC > 15%, debt-to-equity < 0.4, buyback yield between 5–7.5%, and only in the mega-cap space.”

From this shortlist, they can perform deeper quantitative and fundamental analysis.

Shortlisting is fine — it gets investors started.  
However, once I had a shortlist, I found myself asking:  
**Which of these are actually worth researching further?**  
I wanted a **ranked screener based on value**, not just a filter.

---

## The Challenge

Like many retail investors, I don’t have access to Morningstar’s valuations or Bloomberg/FactSet screening tools.  
It’s just me and my Mac.

Recently, I tuned into an old episode of *Value Investing with Legends* featuring **Kim Shannon**, portfolio manager at Sionna Investment Managers.  
She’s used a quantitative screener since 1986 that involves letting *“price = price”* (timestamp 24:04).

What she really meant was:  
> *How does the current share price of a company compare to its disaggregated share price?*

This approach provides a **rough intrinsic value**, letting investors act quickly while identifying warning signs like high leverage or poor asset turnover.

---

## My Approach

Sionna’s screening tool is much more sophisticated, but conceptually we’re doing the same thing.

I **disaggregate price** by:

1. Obtaining a **two-year historical return on equity** using the **DuPont Identity**,  
2. Multiplying that by **equity value per share** to get EPS,  
3. Then multiplying EPS by a **two-year average P/E ratio** to obtain the *disaggregated price*.

*(See: DuPont Identity Explanation.)*

---

## Limitations

There are weaknesses — using historical averages doesn’t capture future changes or business model shifts.  
But that’s not the goal.  
If I wanted the most precise intrinsic values, I’d be running **5000 DCF models a day** and never sleeping.

My goals for the screener are simple:
- Establish a **baseline understanding** of companies  
- Run **quickly**  
- **Save time** — avoid staring at 30 tickers at once  

I also wanted it to be **free** and **easy to use**, even for non-programmers.  
Thanks to **Finviz**, whose [developer API](https://finviz.com/screener.ashx) provides free access to eight quarters of financial data, this became possible.

---

## Implementation

Below is a summary of the key components of the screener, followed by the full Python code.

---

### 1. Install Dependencies

You can run this in **Google Colab** or any local Python environment:

```bash
!pip install finvizfinance
```
```python
"""
This code block creates the universe of companies to analyze.
In my article, I call this shortlisting. Some investors stop here,
but I believe we can go further.
"""
from finvizfinance.screener.valuation import Valuation

# Create a Valuation object
valuation = Valuation()

# Available filters include:
# 'Market Cap.', 'Debt/Equity', 'Return on Investment', etc.
filters_dict = {
    'Market Cap.': '+Mid (over $2bln)',
    'Country': 'USA',
    'Debt/Equity': 'Under 0.3',
    'Return on Investment': 'Over +20%'
}

valuation.set_filter(filters_dict=filters_dict)
df_valuation = valuation.screener_view()
df_valuation
```


```python
import pandas as pd
from finvizfinance.quote import Statements

# Initialize DataFrame
df_financials_summary = pd.DataFrame(columns=[
    'Ticker', 'Price/Share', 'Net Income', 'Total Revenue', 'Total Assets',
    'Total Equity', 'Net Profit Margin (annualized)', 'Asset Turnover',
    'Leverage', 'Normalized ROE', 'Equity/share', 'EPS',
    'Normalized P/E', 'Disaggregated Price', 'Valuation', 'P/FV'
])

# Loop through each ticker
for ticker in df_valuation['Ticker']:
    try:
        stock_statements = Statements()
        df_is = stock_statements.get_statements(ticker, statement='I', timeframe='Q')
        df_bs = stock_statements.get_statements(ticker, statement='B', timeframe='Q')
    except:
        continue

    price = df_valuation.loc[df_valuation['Ticker'] == ticker, 'Price'].values[0]
    net_income, total_revenue, shares_outstanding, price_to_earnings = 0, 0, 0, 0
    assets, equity = 0, 0

    # Use recent data
    is_cols = min(20, len(df_is.columns))
    bs_cols = min(20, len(df_bs.columns))

    # Aggregate historical data
    for i in range(is_cols):
        try:
            net_income += float(df_is.loc['Net Income'].values[i].replace(',', ''))
            total_revenue += float(df_is.loc['Total Revenue'].values[i].replace(',', ''))
            shares_outstanding += float(df_is.loc['Shares Outstanding'].values[i].replace(',', ''))
            price_to_earnings += float(df_is.loc['Price To Earnings Ratio'].values[i].replace(',', ''))
        except:
            continue

    for i in range(bs_cols):
        try:
            assets += float(df_bs.loc['Total Assets'].values[i].replace(',', ''))
            equity += float(df_bs.loc['Total Equity'].values[i].replace(',', ''))
        except:
            continue

    try:
        net_income_avg = net_income / is_cols
        total_revenue_avg = total_revenue / is_cols
        assets_avg = assets / bs_cols
        equity_avg = equity / bs_cols
        shares_outstanding_avg = shares_outstanding / is_cols
    except ZeroDivisionError:
        continue

    # DuPont breakdown
    try:
        net_profit_margin = (net_income_avg * 4) / total_revenue_avg
        asset_turnover = total_revenue_avg / assets_avg
        leverage = assets_avg / equity_avg
    except ZeroDivisionError:
        continue

    roe = net_profit_margin * asset_turnover * leverage
    equity_per_share = equity_avg / shares_outstanding_avg
    eps = roe * equity_per_share

    # Disaggregated Price
    price_to_earnings_avg = price_to_earnings / is_cols
    disaggregated_price = eps * price_to_earnings_avg

    new_row = pd.DataFrame([[
        ticker, price, int(net_income_avg)/100, int(total_revenue_avg)/100,
        int(assets_avg)/100, int(equity_avg)/100, net_profit_margin,
        asset_turnover, leverage, roe, equity_per_share, eps,
        price_to_earnings_avg, round(disaggregated_price, 2),
        'Overvalued' if (price > disaggregated_price * 1.05)
        else 'Undervalued' if (price < disaggregated_price * 0.95)
        else 'Fair Valued',
        round(price / disaggregated_price, 2)
    ]], columns=df_financials_summary.columns)

    if not new_row.isnull().all().all():
        df_financials_summary = pd.concat([df_financials_summary, new_row], ignore_index=True)

# Save results
df_financials_summary.to_csv('screen.csv')
df_financials_summary
```