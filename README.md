# Setup Instructions

## Pre-requisites

1. Have docker installed for running Milvus Vector DB with ease.

2. Have [uv](https://docs.astral.sh/uv/) python package manager.

## Steps

1. Create virtual environment with uv and then activate the environment.

```
uv venv --python 3.13
```

2. Activate virtual environment.

For MacOS and Linux:
```
source .venv/bin/activate
```

For Windows:
```
.venv\Scripts\activate
```

3. Setup environment variables by creating .env based on the [.env.example](.env.example). Replace API keys with the actual keys (Gemini API key) to run the code.

4. Run Milvus vector DB
```
docker compose up -d
```

5. Upsert data to vector DB
```
python setup.py
```

6. Run and test the agent
```
python main.py
```

If there are any troubles during the setup of the project, you can contact me via lequan310.official@gmail.com
