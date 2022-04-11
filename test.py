import requests
sample_request_input = {'body': 'un test quelconque de the english', 'title': 'principal'}
response = requests.get("http://localhost:8000/predict", json=sample_request_input)
print(response.text)