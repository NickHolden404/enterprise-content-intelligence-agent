# Deployment & Usage Guide
## Enterprise Content Intelligence Agent

This guide covers deployment options, configuration, and usage examples for the Enterprise Content Intelligence Agent.

---

## 🚀 Quick Start

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/enterprise-agent.git
cd enterprise-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the agent
python main.py
```

### Environment Variables

Create a `.env` file with the following:

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional
DATABASE_CONNECTION=postgresql://user:pass@localhost:5432/dbname
LOG_LEVEL=INFO
ENABLE_TRACING=true
PORT=8080

# MCP Configuration
MCP_FILESYSTEM_ROOT=/path/to/documents
MCP_GIT_REPOS=/path/to/repos

# Search Configuration
GOOGLE_SEARCH_API_KEY=your_search_api_key
GOOGLE_SEARCH_CX=your_search_cx
```

---

## 🐳 Docker Deployment

### Build Container

```bash
# Build the Docker image
docker build -t enterprise-agent:latest .

# Run locally
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_key \
  -e LOG_LEVEL=INFO \
  enterprise-agent:latest
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DATABASE_CONNECTION=postgresql://postgres:password@db:5432/enterprise
      - LOG_LEVEL=INFO
    depends_on:
      - db
    volumes:
      - ./data:/app/data
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=enterprise
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
```

Run with:
```bash
docker-compose up -d
```

---

## ☁️ Cloud Run Deployment (Google Cloud)

### Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Deploy to Cloud Run

```bash
# 1. Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/enterprise-agent

# 2. Deploy to Cloud Run
gcloud run deploy enterprise-agent \
  --image gcr.io/YOUR_PROJECT_ID/enterprise-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --max-instances 10

# 3. Get service URL
gcloud run services describe enterprise-agent \
  --region us-central1 \
  --format 'value(status.url)'
```

### Deploy with Agent Engine

```bash
# Install Agent Engine CLI
pip install google-cloud-agent-engine

# Initialize configuration
agent-engine init --project YOUR_PROJECT_ID

# Deploy agent
agent-engine deploy \
  --source main.py \
  --name enterprise-agent \
  --region us-central1 \
  --env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY
```

---

## 🔧 Configuration

### MCP Server Setup

Configure MCP servers for tool access:

```python
# config.yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"]
    env:
      ALLOWED_PATHS: "/enterprise/documents,/shared/resources"
  
  git:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-git"]
    env:
      GIT_REPOS: "/repos/main-app,/repos/data-pipeline"
  
  postgres:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "postgresql://user:pass@localhost:5432/db"
```

### Custom Tools

Add custom enterprise tools:

```python
# tools/custom_tools.py
from google.generativeai import tool

@tool
def query_crm_system(customer_id: str) -> dict:
    """Query customer data from CRM system."""
    # Your CRM integration logic
    return {
        'customer_id': customer_id,
        'name': 'Acme Corp',
        'status': 'active'
    }

@tool
def search_slack_history(query: str, channels: list) -> list:
    """Search Slack message history."""
    # Your Slack integration logic
    return [
        {'message': 'Relevant message 1', 'channel': '#engineering'},
        {'message': 'Relevant message 2', 'channel': '#product'}
    ]
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Configure `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'enterprise-agent'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Access Metrics

```bash
# Query metrics
curl http://localhost:8080/metrics

# Example metrics available:
# - agent_queries_total
# - agent_query_duration_seconds
# - agent_active_count
# - agent_errors_total
```

### Cloud Trace

View distributed traces in Google Cloud Console:

```bash
# View traces
gcloud trace list --project YOUR_PROJECT_ID

# Filter traces
gcloud trace list --filter="name:orchestrator.process_query"
```

### Structured Logs

Query logs with structured fields:

```bash
# Google Cloud Logging
gcloud logging read "resource.type=cloud_run_revision" \
  --limit 50 \
  --format json

# Filter by severity
gcloud logging read "severity>=ERROR"

# Filter by custom fields
gcloud logging read "jsonPayload.agent_name=WebSearchAgent"
```

---

## 💻 Usage Examples

### Python Client

```python
import requests

# Configure endpoint
AGENT_URL = "http://localhost:8080"  # or your Cloud Run URL

# Process query
response = requests.post(
    f"{AGENT_URL}/query",
    json={
        "query": "Analyze our Q3 sales data",
        "user_id": "john.doe@company.com"
    }
)

result = response.json()
print(f"Synthesis: {result['synthesis']['content']}")
print(f"Confidence: {result['synthesis']['confidence']}")
print(f"Agents Used: {result['synthesis']['agents_used']}")
```

### Multi-turn Conversation

```python
# Initial query
response1 = requests.post(
    f"{AGENT_URL}/query",
    json={
        "query": "What are the latest AI security threats?",
        "user_id": "jane@company.com"
    }
)

session_id = response1.json()['session_id']

# Follow-up with context
response2 = requests.post(
    f"{AGENT_URL}/query",
    json={
        "query": "How do these affect our cloud infrastructure?",
        "user_id": "jane@company.com",
        "session_id": session_id
    }
)

# Agent maintains context from first query
print(response2.json()['synthesis']['content'])
```

### Batch Processing

```python
import asyncio
import aiohttp

async def process_queries(queries):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in queries:
            task = session.post(
                f"{AGENT_URL}/query",
                json={"query": query, "user_id": "batch_user"}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# Process multiple queries concurrently
queries = [
    "Summarize Q3 performance",
    "Find API documentation",
    "Analyze customer feedback"
]

results = asyncio.run(process_queries(queries))
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=main --cov-report=html

# Run specific test category
pytest tests/test_agent_evaluation.py -v

# Run performance benchmarks
pytest tests/ -m benchmark -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8080
```

Example `load_test.py`:

```python
from locust import HttpUser, task, between

class AgentUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def query_agent(self):
        self.client.post("/query", json={
            "query": "test query",
            "user_id": f"user_{self.user_id}"
        })
```

---

## 🔐 Security

### API Authentication

Implement authentication for production:

```python
# Add to main.py
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'No token provided'}, 401
        
        try:
            jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/query', methods=['POST'])
@require_auth
def process_query():
    # Handler code
    pass
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/query')
@limiter.limit("10 per minute")
def process_query():
    pass
```

### Secrets Management

```bash
# Google Secret Manager
echo -n "your-api-key" | gcloud secrets create GOOGLE_API_KEY --data-file=-

# Access in Cloud Run
gcloud run deploy enterprise-agent \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest
```

---

## 📈 Scaling

### Horizontal Scaling

```bash
# Cloud Run auto-scaling
gcloud run services update enterprise-agent \
  --min-instances 1 \
  --max-instances 100 \
  --concurrency 80
```

### Performance Optimization

```python
# Enable caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_memory(query: str):
    return memory_agent.execute(query)

# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. API Key Issues**
```bash
# Verify API key is set
echo $GOOGLE_API_KEY

# Test API access
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('OK')"
```

**2. MCP Connection Errors**
```bash
# Check MCP server status
npx @modelcontextprotocol/inspector

# Verify permissions
ls -la /path/to/documents
```

**3. Memory Issues**
```bash
# Increase memory limit
docker run -m 4g enterprise-agent

# Monitor memory usage
docker stats
```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python main.py

# Enable trace debugging
ENABLE_TRACING=true python main.py
```

### Health Checks

```bash
# Check service health
curl http://localhost:8080/health

# Check agent status
curl http://localhost:8080/status
```

---

## 📚 Additional Resources

- [Google Gemini Documentation](https://ai.google.dev/docs)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Agent Engine Documentation](https://cloud.google.com/agent-engine/docs)

---

## 🤝 Support

For issues or questions:

1. Check the [troubleshooting guide](#-troubleshooting)
2. Review [GitHub Issues](https://github.com/yourusername/enterprise-agent/issues)
3. Contact: [your-email@example.com]

---

## 📄 License

MIT License - See LICENSE file for details
