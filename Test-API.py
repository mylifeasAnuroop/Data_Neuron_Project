import requests

url = 'https://data-neuron-project.uc.r.appspot.com'
data = {
    "text1": "nuclear body seeks new tech .......",
    "text2": "terror suspects face arrest ......"
}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Similarity Score:", result["similarity_score"])
else:
    print("Error:", response.text)
