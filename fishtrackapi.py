# import requests
# api = 'https://gateway.api.globalfishingwatch.org/v2/vessels/search?query=7831410&datasets=public-global-fishing-vessels:latest&limit=1&offset=0' \
#   -H "Authorization: Bearer [TOKEN]'
# def get_data(self, api):
#     response = requests.get(f"{api}")
#     if response.status_code == 200:
#         print("sucessfully fetched the data")
#         self.formatted_print(response.json())
#     else:
#         print(f"Hello person, there's a {response.status_code} error with your request")


import requests

# Define the API endpoint URL
url = 'https://gateway.api.globalfishingwatch.org/v2/vessels/search?query=7831410&datasets=public-global-fishing-vessels:latest&limit=1&offset=0'  # Replace with the actual API URL

# Define your bearer token
bearer_token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJmaXNodHJhY2tpbmciLCJ1c2VySWQiOjI4NzM4LCJhcHBsaWNhdGlvbk5hbWUiOiJmaXNodHJhY2tpbmciLCJpZCI6MTAyMywidHlwZSI6InVzZXItYXBwbGljYXRpb24ifSwiaWF0IjoxNjk4MzA1MTkwLCJleHAiOjIwMTM2NjUxOTAsImF1ZCI6ImdmdyIsImlzcyI6ImdmdyJ9.fdDfsaINgBFfhUGkDoBrC_TsO-dktSCpMZql4DKww83m4nJmeItj8nGfEMebCQhjM8eOJ6ML0y2k80hJ8BNf4CkgIyLJjtPdpyHLs-2TNCWVQ-5A8ryNtKC8GeAx6Swisy2FgZ454gdguK_XRcpCffoF0k9RlO0nbg9Jfv98ku1R15AVTSyY7ziAWV4BIqnz8Au75ms1odp168vAZR0V30Udx5OdoWMO0-fQjKnSLGkGjGfrF-WOuW6l0uRT79Ghq8ScJ0OzTmGf5TN4FvxiiyMetfrPtBTxHwlQ4VsXLfaApFLYhcYRgABqqRybmiUSm0xAGqizyUu-G2179HLRLffyeZD-RH9CoeLa2vZ85F0n7Ufaf_uy5HjGG1tUYZEElCp1ka3jRsm3WC4LZp-AdZFpr2gwTttUFauS3Vb2xi-tFqmXJjZzRWNAX3rqUhlbCVcSWjA9HhG2eX6bJxFY-0dIRp0Dhpg5_t4lsYXqUNxhMT7iWx8qNC_Ry6dYlYxB'

# Set up the headers with the bearer token
headers = {
    'Authorization': f'Bearer {bearer_token}',
    'Content-Type': 'application/json'  # You may need to adjust this based on the API's requirements
}

# Make the GET request to the API
response = requests.get(url, headers=headers)

# Check the response status code
if response.status_code == 200:
    # Request was successful, and you can work with the API data
    api_data = response.json()  # If the API returns JSON data
    print(api_data)
else:
    # Request failed
    print(f"Request failed with status code {response.status_code}")
    print(response.text)