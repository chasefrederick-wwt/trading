#create virtual environment if not done so already
python3 -m venv venv 

#activate virtual environment
source venv/bin/activate 

#install dependencies
pip install -r requirements.txt 

#run the script
python3 trade.py 

2. Get a News API key:
   - Go to https://newsapi.org/
   - Register for a free account
   - Get your API key from the dashboard

3. Set up your API key:
   - Copy the `.env.example` file to `.env`
   - Replace `your-api-key-here` with your actual News API key

## Usage

Run the script:

```bash
python3 trade.py
```

The script will generate trading suggestions based on:
- Technical indicators (RSI, volatility)
- News sentiment
- Options chain data

## Notes

- The free tier of News API has a limit of 100 requests per day
- Without a valid API key, the system will still work but without news sentiment analysis
