import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GROQ_API_KEY')
res = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {key}'})
data = res.json()
if 'data' in data:
    for m in data['data']:
        print(m['id'])
else:
    print(data)
