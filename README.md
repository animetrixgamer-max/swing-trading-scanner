# Indian Swing Scanner — Mobile App

A free Streamlit-based mobile-first web app for the swing-trading strategy developed in this conversation.

## What it does

- Default universe: NIFTY 200
- Optional NIFTY 50
- Optional custom NSE symbols
- Default capital: ₹20,000
- Default risk: 1%
- Candidate threshold: Score >= 80
- Buy threshold: Score >= 90 and all Stage-1 technical conditions pass plus bullish NIFTY
- Developing candidates remain visible with failed conditions
- Stage 2 earnings check runs only for actual Stage-1 buy candidates
- 5 years of daily Yahoo Finance OHLCV data
- Calculates 20 EMA, 50 EMA, 200 SMA, RSI(14), ATR(14), previous 20D high, 20D average volume and 52W high locally
- Calculates stop, quantity, ₹ risk and 2R target
- Downloads Excel files with the complete scan and the final candidate sheet
- Optional Google Drive upload via Google Cloud service-account credentials

## Free deployment

Streamlit Community Cloud is free and deploys from GitHub. Create a GitHub repository, upload `app.py`, `requirements.txt`, and `.streamlit/config.toml`, then deploy the app from Streamlit Community Cloud.

## Google Drive upload

The scanner works without Drive upload. The app always lets you download the Excel files directly.

To enable Drive upload:

1. Create a Google Cloud project.
2. Enable the Google Drive API.
3. Create a service account and a JSON key.
4. Create a Drive folder and share that folder with the service-account email with Editor access.
5. In Streamlit Community Cloud, add a secret named `[gcp_service_account]` containing the service-account JSON fields.
6. Paste the target Drive folder ID in the app.

For a long-term setup, a Google user OAuth flow can also be used, but the service-account folder method is simpler for a private personal app.

## Important data note

This app uses Yahoo Finance through `yfinance` for the automated raw OHLCV feed. It does not invent missing market data. If data cannot be downloaded or indicators cannot be calculated, that stock is excluded as data-unavailable.

Yahoo Finance is not the NSE/BSE exchange feed. Verify any final trade with NSE/BSE/broker data and official corporate announcements.
