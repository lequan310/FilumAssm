# Filum Pain Point to Solution Agent

## 🚀 Quick Start

### Prerequisites

Before getting started, ensure you have the following installed:

- **Docker** - For running Milvus Vector DB
- **[uv](https://docs.astral.sh/uv/)** - Python package manager

### Installation

1. **Create and activate virtual environment**
   ```bash
   uv venv --python 3.13
   ```

2. **Activate the environment**
   
   **MacOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```
   
   **Windows:**
   ```cmd
   .venv\Scripts\activate
   ```

3. **Configure environment variables**
   
   Create a `.env` file based on [.env.example](.env.example) and add your API keys:
   ```bash
   cp .env.example .env
   ```
   > ⚠️ **Important:** Replace placeholder values with your actual API key

### Running the Application

1. **Start Milvus Vector DB**
   ```bash
   docker compose up -d
   ```

2. **Initialize the database**
   ```bash
   python setup.py
   ```

3. **Run the agent**
   ```bash
   python main.py
   ```

## 🧹 Cleanup

To stop and remove the Milvus vector database:
```bash
docker compose down
```

## 📧 Support

If you encounter any issues during setup, feel free to reach out:
- **Email:** lequan310.official@gmail.com

## 📁 Project Structure

```
FilumAssm/
├── .github                 # GitHub Actions
├── agent                   # Agent Implementation
├── data                    # Example data to upsert into Milvus
├── .env.example            # Environment variables template
├── docker-compose.yml      # Milvus database configuration
├── embedding.py            # Embedding function
├── main.py                 # Main application entry point
├── milvus_connector.py     # Milvus class for creating collection, upsert documents, and hybrid search
├── milvus.yaml             # Milvus file config
├── setup.py                # Database initialization script
├── .gitignore              # Gitignore
├── pyproject.toml          # Dependencies
├── ruff.toml               # Gitignore
├── uv.lock                 # uv lock file
└── README.md               # This file
```
