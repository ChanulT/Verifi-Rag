# Embedding Service

A production-ready microservice for generating text embeddings using sentence-transformers with GPU acceleration and dynamic configuration management.

## 🎯 Features

- **GPU-Accelerated Embeddings**: Automatic GPU/CPU offloading for optimal performance
- **Dynamic Configuration**: Runtime configuration updates via database without restart
- **Hot-Reload**: Change embedding models without service downtime
- **Enterprise-Ready**: Thread-safe operations, comprehensive metrics, health checks
- **Scalable Architecture**: Repository pattern, dependency injection, async-first design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Application                  │
├─────────────────────────────────────────────────────────┤
│  Routes Layer        │  Business Logic    │  Data Layer │
│  (routes.py)         │  (service.py)      │ (database.py)│
├──────────────────────┼────────────────────┼──────────────┤
│  - API Endpoints     │  - Model Manager   │  - Repository│
│  - Request Validation│  - Embeddings      │  - ORM Models│
│  - Response Mapping  │  - Metrics         │  - Migrations│
└─────────────────────────────────────────────────────────┘
              ▲                    ▲                ▲
              └────────────────────┴────────────────┘
                    Configuration Layer (config.py)
```

## 📋 Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, falls back to CPU)
- MySQL/MariaDB (optional, for database-backed configuration)

## 🚀 Quick Start

### 1. Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configuration

Create a `.env` file:

```env
# Service Configuration
EMBEDDING_HOST=0.0.0.0
EMBEDDING_PORT=8001

# Model Configuration
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cuda  # or 'cpu' or 'auto' for GPU-CPU offloading
EMBEDDING_MODEL_CACHE_DIR=./model_cache
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_BATCH_SIZE=32

# Database Configuration (optional)
EMBEDDING_USE_DB_SETTINGS=false
EMBEDDING_DB_URL=mysql+aiomysql://user:pass@localhost:3306/embedding_db

# CORS
CORS_ORIGINS=*
```

### 3. Run the Service

```bash
# Development mode with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
```

## 📡 API Endpoints

### Embeddings

**POST `/embed`** - Generate embeddings for multiple texts
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Machine learning is amazing"]
  }'
```

**POST `/embed/single`** - Generate embedding for single text
```bash
curl -X POST "http://localhost:8001/embed/single?text=Hello%20world"
```

### Health & Monitoring

**GET `/health`** - Service health check
**GET `/metrics`** - Service metrics
**GET `/model/info`** - Model information

### Configuration Management

**GET `/settings`** - Get current settings
**POST `/settings`** - Update a setting
**POST `/settings/bulk`** - Bulk update settings
**DELETE `/settings/{key}`** - Delete a setting (revert to default)

## 🔧 Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_name` | str | `BAAI/bge-small-en-v1.5` | HuggingFace model identifier |
| `device` | str | `cuda` | Device: `cuda`, `cpu`, or `auto` |
| `model_cache_dir` | str | `./model_cache` | Model cache directory |
| `normalize_embeddings` | bool | `true` | L2 normalize embeddings |
| `max_batch_size` | int | `32` | Maximum batch size |
| `host` | str | `0.0.0.0` | Service host |
| `port` | int | `8001` | Service port |
| `cors_origins` | list | `["*"]` | CORS allowed origins |

## 🎨 Design Patterns

### Repository Pattern
- Clean separation between data access and business logic
- Testable and maintainable data operations

### Dependency Injection
- FastAPI's built-in DI for database sessions and dependencies
- Loose coupling between components

### Thread-Safe Singleton
- Configuration manager with lock-based synchronization
- Safe concurrent access to settings

### Async-First Architecture
- Non-blocking I/O operations
- Efficient handling of concurrent requests

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## 📊 Performance Considerations

- **GPU Offloading**: Use `device=auto` for automatic GPU-CPU memory management
- **Batch Size**: Adjust `max_batch_size` based on available GPU memory
- **Connection Pooling**: Database connections are pooled (5 base + 10 overflow)
- **Model Caching**: Models are cached locally to avoid repeated downloads

## 🔒 Security Best Practices

- ✅ Environment-based configuration (no hardcoded secrets)
- ✅ SQL injection protection via SQLAlchemy ORM
- ✅ Input validation using Pydantic
- ✅ CORS configuration
- ✅ Health check endpoints for monitoring

## 🐳 Docker Deployment

```dockerfile
# See Dockerfile for production deployment
docker build -t embedding-service .
docker run -p 8001:8001 --gpus all embedding-service
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue.