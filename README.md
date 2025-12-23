### Setting up the Virtual Environment (.venv)

```bash
python3 -m venv .venv
```

## Activating the Virtual Environment

- Windows:

```bash
.\.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Deactivating the Environment

```bash
deactivate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Freeze dependencies

```bash
pip freeze > requirements.txt
```
