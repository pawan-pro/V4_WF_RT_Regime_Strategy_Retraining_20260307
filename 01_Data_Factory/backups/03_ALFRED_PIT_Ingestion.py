# Databricks notebook source
import pandas as pd
import numpy as np

# 1. Define Paths
input_path = "/Volumes/pawan_quant_research_1/default/market_data/filtered_economic_indicators.csv"
output_path = "/Volumes/pawan_quant_research_1/default/market_data/economic_indicators_with_revisions.csv"

# 2. Load the data
df = pd.read_csv(input_path)

# 3. Pre-processing: Convert time and handle duplicates
df['time'] = pd.to_datetime(df['time'])

# Normalize Event Names: 
# This removes prefixes like "United States-Employment-" so that 
# "Non-Farm Payrolls" and "United States-Employment-Non-Farm Payrolls" match.
df['event_clean'] = df['event'].str.replace('United States-', '', case=False)
df['event_clean'] = df['event_clean'].str.split('-').str[-1].str.strip()

# 4. Sort by event and time
# This ensures that the report from February follows the report from January
df = df.sort_values(by=['event_clean', 'time'])

# 5. Identify Revisions using the "Chain-Link" method
# Logic: We look at the 'actual' value from the PREVIOUS report (shift 1) 
# and compare it to the 'prev' value stated in the CURRENT report.
df['last_month_actual'] = df.groupby('event_clean')['actual'].shift(1)

# A revision is detected if 'prev' in current row != 'actual' in previous row
# We use np.isclose to handle small floating point differences
df['was_revised'] = ~np.isclose(df['prev'], df['last_month_actual'], equal_nan=True) & df['last_month_actual'].notna()

# Calculate the magnitude of the revision
df['revision_delta'] = df['prev'] - df['last_month_actual']

# 6. Formatting for the final CSV
# Create a descriptive column for clarity
df['revision_note'] = ""
df.loc[df['was_revised'], 'revision_note'] = (
    "Original: " + df['last_month_actual'].astype(str) + 
    " -> Revised to: " + df['prev'].astype(str)
)

# 7. Save the processed file back to the Volume
# We keep the original columns and add our new analysis columns
df.to_csv(output_path, index=False)

print(f"File saved successfully to: {output_path}")

# Display a sample of the revisions found
display(df[df['was_revised'] == True][['time', 'event_clean', 'last_month_actual', 'prev', 'revision_delta']].head(10))