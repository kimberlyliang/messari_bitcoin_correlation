# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# Define the path to the folder containing the CSV files
csv_folder_path = '/Users/kimberlyliang/Documents/Bitcoin Data Analysis/All asset csv files/'

# Helper function to read CSV with flexible date parsing
def load_csv_with_date_handling(file_path):
    with open(file_path, 'r') as f:
        headers = next(f).strip().split(',')
        date_col_index = next(i for i, col in enumerate(headers) if col.lower() == 'date')
        
        for line in f:
            row = line.strip().split(',')
            try:
                row[date_col_index] = pd.to_datetime(row[date_col_index])
                if 'ASSET_CIRCULATING_MARKETCAP_USD' in headers:
                    marketcap_index = headers.index('ASSET_CIRCULATING_MARKETCAP_USD')
                    if float(row[marketcap_index]) <= 10_000_000:
                        continue
                yield {headers[i]: row[i] for i in range(len(headers))}
            except ValueError:
                continue

def process_data_in_chunks(file_path, chunk_size=10000):
    data = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.strip()
        date_col = next((col for col in chunk.columns if col.lower() == 'date'), None)
        
        if date_col is None:
            raise ValueError(f"No date column found in {file_path}")
        
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
        chunk = chunk.dropna(subset=[date_col])
        
        if 'ASSET_CIRCULATING_MARKETCAP_USD' in chunk.columns:
            chunk = chunk[chunk['ASSET_CIRCULATING_MARKETCAP_USD'] > 10_000_000]
        
        chunk = chunk.rename(columns={date_col: 'DATE'})
        data.append(chunk)
    
    return pd.concat(data)

# Load the datasets
btc_usd = process_data_in_chunks(os.path.join(csv_folder_path, 'bitcoin.csv'))
eth_usd = process_data_in_chunks(os.path.join(csv_folder_path, 'ETH-USD.csv'))
bnb_usd = process_data_in_chunks(os.path.join(csv_folder_path, 'BNB-USD.csv'))
sol_usd = process_data_in_chunks(os.path.join(csv_folder_path, 'SOL-USD.csv'))
spy = process_data_in_chunks(os.path.join(csv_folder_path, 'SPY.csv'))
nvda = process_data_in_chunks(os.path.join(csv_folder_path, 'NVDA.csv'))
oil = process_data_in_chunks(os.path.join(csv_folder_path, 'Oil!.csv'))
gold = process_data_in_chunks(os.path.join(csv_folder_path, 'Gold!.csv'))
nasdaq = process_data_in_chunks(os.path.join(csv_folder_path, 'Nasdaq!.csv'))
energy_networks = process_data_in_chunks(os.path.join(csv_folder_path, 'Energy Networks.csv'))
layer_0 = process_data_in_chunks(os.path.join(csv_folder_path, 'Layer 0.csv'))
smart_contract = process_data_in_chunks(os.path.join(csv_folder_path, 'Smart Contract Platform.csv'))
lending = process_data_in_chunks(os.path.join(csv_folder_path, 'Lending.csv'))
exchange = process_data_in_chunks(os.path.join(csv_folder_path, 'Exchange.csv'))
social_tokens = process_data_in_chunks(os.path.join(csv_folder_path, 'Social Tokens.csv'))
gaming = process_data_in_chunks(os.path.join(csv_folder_path, 'Gaming.csv'))
real_world_assets = process_data_in_chunks(os.path.join(csv_folder_path, 'Real World Assets.csv'))

def get_top_20_assets(df):
    if 'ASSET_CIRCULATING_MARKETCAP_USD' in df.columns:
        latest_data = df.loc[df['DATE'] == df['DATE'].max()]
        top_20 = latest_data.nlargest(20, 'ASSET_CIRCULATING_MARKETCAP_USD')
        return top_20['ASSET_SYMBOL'].tolist()
    else:
        return df.columns.tolist()[:20]  # For non-crypto assets, return all columns (up to 20)

def create_multiindex_dataframe(data_dict):
    dfs = []
    for asset, df in data_dict.items():
        top_20_assets = get_top_20_assets(df)
        
        if 'ASSET_CLOSE_VWAP_USD' in df.columns:
            df = df[df['ASSET_SYMBOL'].isin(top_20_assets)]
            # Sort the dataframe and keep the last occurrence of each duplicate
            df = df.sort_values('DATE').groupby(['DATE', 'ASSET_SYMBOL']).last().reset_index()
            df = df.pivot(index='DATE', columns='ASSET_SYMBOL', values='ASSET_CLOSE_VWAP_USD')
        else:
            if 'Adj Close' in df.columns:
                price_col = 'Adj Close'
            elif 'Close' in df.columns:
                price_col = 'Close'
            elif 'Close/Last' in df.columns:
                price_col = 'Close/Last'
            else:
                raise ValueError(f"No suitable close price column found for {asset}")
            
            df = df[['DATE', price_col]].rename(columns={price_col: asset})
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')
        
        df = df[~df.index.duplicated(keep='last')]
        dfs.append(df)
    
    start_date = pd.to_datetime('2018-01-01')
    latest_date = max(df.index.max() for df in dfs)
    date_range = pd.date_range(start=start_date, end=latest_date)
    
    dfs = [df.reindex(date_range) for df in dfs]
    merged_data = pd.concat(dfs, axis=1)
    merged_data = merged_data.ffill().dropna()
    
    return merged_data

data_dict = {
    'BTC': btc_usd,
    'ETH': eth_usd,
    'BNB': bnb_usd,
    'SOL': sol_usd,
    'SPY': spy,
    'NVDA': nvda,
    'Oil': oil,
    'Gold': gold,
    'Nasdaq': nasdaq,
    'Energy': energy_networks,
    'Layer0': layer_0,
    'SmartContract': smart_contract,
    'Lending': lending,
    'Exchange': exchange,
    'SocialTokens': social_tokens,
    'Gaming': gaming,
    'RealWorldAssets': real_world_assets
}

# Print column names for debugging
for asset, df in data_dict.items():
    print(f"Columns in {asset} dataframe:")
    print(df.columns.tolist())

# Merging datasets on 'DATE' using suffixes to differentiate overlapping columns
merged_data = create_multiindex_dataframe(data_dict)

# Check for missing values
print(merged_data[['BTC', 'ETH', 'BNB', 'SOL', 'SPY', 'NVDA', 'Oil', 'Gold', 'Nasdaq']].isnull().sum())

# Calculate percentage change for all assets
for column in merged_data.columns:
    merged_data[f'{column}_pct_change'] = merged_data[column].pct_change()

# Calculating the rolling correlation with Bitcoin (90-day window)
rolling_window = 90

correlations = {'Date': merged_data.index}
for column in merged_data.columns:
    if column != 'BTC' and not column.endswith('_pct_change'):
        correlations[column] = merged_data['BTC_pct_change'].rolling(window=rolling_window).corr(merged_data[f'{column}_pct_change'])

# Create a DataFrame from the correlations dictionary
corr_df = pd.DataFrame(correlations).set_index('Date')

# Save the correlation DataFrame to a CSV file
corr_df.to_csv('bitcoin_correlations.csv')
print("Correlation data has been saved to 'bitcoin_correlations.csv'")

# Plot correlations for BTC, ETH, SOL, BNB
plt.figure(figsize=(14, 10))
for asset in ['ETH', 'SOL', 'BNB']:
    plt.plot(corr_df.index, corr_df[asset], label=asset, linewidth=1.5)

plt.title('Rolling Correlation with Bitcoin - Major Cryptocurrencies (90-Day Window)')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlation_plot_major_crypto.png')
plt.close()
print("Correlation plot for major cryptocurrencies has been saved to 'correlation_plot_major_crypto.png'")

# Plot correlations for BTC and traditional finance assets
plt.figure(figsize=(14, 10))
for asset in ['SPY', 'Gold', 'Oil', 'Nasdaq']:
    plt.plot(corr_df.index, corr_df[asset], label=asset, linewidth=1.5)

plt.title('Rolling Correlation with Bitcoin - Traditional Finance Assets (90-Day Window)')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlation_plot_traditional_finance.png')
plt.close()
print("Correlation plot for traditional finance assets has been saved to 'correlation_plot_traditional_finance.png'")

# Plot correlations for BTC and crypto categories
crypto_categories = ['Energy', 'Layer0', 'SmartContract', 'Lending', 'Exchange', 'SocialTokens', 'Gaming', 'RealWorldAssets']
for i in range(0, len(crypto_categories), 4):
    plt.figure(figsize=(14, 10))
    for category in crypto_categories[i:i+4]:
        category_columns = [col for col in corr_df.columns if col.startswith(category)]
        avg_correlation = corr_df[category_columns].mean(axis=1)
        plt.plot(corr_df.index, avg_correlation, label=category, linewidth=1.5)

    plt.title(f'Rolling Correlation with Bitcoin - Crypto Categories (Set {i//4 + 1}) (90-Day Window)')
    plt.xlabel('Date')
    plt.ylabel('Average Correlation')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'correlation_plot_crypto_categories_set{i//4 + 1}.png')
    plt.close()
    print(f"Correlation plot for crypto categories (Set {i//4 + 1}) has been saved to 'correlation_plot_crypto_categories_set{i//4 + 1}.png'")

# Plot closing prices for all assets
plt.figure(figsize=(16, 10))

assets_to_plot = ['BTC', 'ETH', 'BNB', 'SOL', 'SPY', 'NVDA', 'Oil', 'Gold', 'Nasdaq']

for asset in assets_to_plot:
    plt.plot(merged_data.index, merged_data[asset], label=asset, linewidth=1.5)

plt.title('Closing Prices of All Assets')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Improve x-axis date formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('all_assets_prices.png')
plt.close()
print("All assets price plot has been saved to 'all_assets_prices.png'")

# Check for missing values
print(merged_data[['BTC', 'ETH', 'BNB', 'SOL', 'SPY', 'NVDA', 'Oil', 'Gold', 'Nasdaq']].isnull().sum())

# Save the merged data with correlations to a CSV file
merged_data.to_csv('merged_data_with_correlations.csv')
print("Merged data with correlations has been saved to 'merged_data_with_correlations.csv'")