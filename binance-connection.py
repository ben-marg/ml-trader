import json


with open('keys.json', 'r') as file:
    credentials = json.load(file)

client = UMFutures(binance_config.get('apiKey'), binance_config.get('secret'), base_url = 'https://testnet.binancefuture.com' )

response = client.balance(recvWindow=6000)
print(response)