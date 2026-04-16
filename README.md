# End-to-End Multi-PDF RAG Application (FastAPI + React + pgvector + Redis)

This repository provides a production-style baseline for:
- PDF upload to **S3**
- Text chunking + embeddings
- Vector retrieval using **Postgres + pgvector**
- Metadata filtering (example: `category`)
- **JWT authentication**
- **Streaming responses** from LLM endpoint
- **Redis cache** for repeated questions
- Multi-PDF query support (`document_ids`)

## 1) Local Setup

```bash
cp .env.example .env
docker compose up --build
```

Then initialize admin:

```bash
curl -X POST http://localhost:8005/auth/bootstrap-admin
```

Open UI: `http://localhost:5173`

## 2) Architecture

1. User logs in (JWT token from `/auth/login`).
2. User uploads one/many PDFs from UI.
3. Backend stores original file in S3.
4. Backend extracts text using `pypdf`, splits with LangChain splitter.
5. Backend calls embedding service (`/embed`) and stores vectors in `document_chunks.embedding` (pgvector).
6. User asks query.
7. Backend checks Redis cache by `(user, query, filters)`.
8. On cache miss, top-K chunks are retrieved using pgvector distance + metadata filters.
9. Prompt is sent to LLM endpoint (`OLLAMA_URL`) with stream enabled.
10. Tokens are emitted as SSE to frontend.

## 3) API Endpoints

- `POST /auth/bootstrap-admin`
- `POST /auth/login`
- `POST /documents/upload` (multipart, supports metadata like category)
- `GET /documents`
- `POST /chat/stream` (SSE streaming)

## 4) AWS Deployment with Docker + GitHub (Step-by-step)

### Step A: Create GitHub repository
1. Create a new repo in GitHub.
2. Push code:
   ```bash
   git init
   git remote add origin <your-github-repo-url>
   git add .
   git commit -m "Initial RAG app"
   git push -u origin main
   ```

### Step B: Provision AWS infrastructure
1. Create **ECR** repositories:
   - `rag-api`
   - `rag-frontend`
2. Create **RDS PostgreSQL** instance and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Create **ElastiCache Redis** cluster.
4. Create **S3 bucket** (private).
5. Create IAM user/role with access to ECR + S3.
6. Store secrets in **AWS Secrets Manager** or **SSM Parameter Store**.

### Step C: Deploy target
Use one of these:
- **ECS Fargate** (recommended)
- EC2 with Docker Compose

For ECS Fargate:
1. Create ECS cluster.
2. Create task definitions for `api` and `frontend`.
3. Add environment vars/secrets from Secrets Manager.
4. Attach ALB:
   - `/api/*` -> api service
   - `/` -> frontend service
5. Configure security groups for ECS -> RDS/Redis access.

### Step D: GitHub Actions CI/CD
Create workflow to build and push images to ECR on every push to `main`.
Then trigger ECS service update.

Minimal flow:
1. Checkout code.
2. Configure AWS credentials.
3. Login to ECR.
4. Build/push `backend/Dockerfile` and `frontend` container.
5. Update ECS service (`aws ecs update-service --force-new-deployment`).

### Step E: Production hardening
- Use HTTPS via ACM + ALB.
- Add WAF rules.
- Rotate JWT secret and API keys.
- Enable CloudWatch logs and alarms.
- Add DB backups and retention policies.
- Add request rate limits.

## 5) Important Notes
- Do not commit real secrets.
- Replace default admin credentials in production.
- Increase worker count and add queueing (Celery/SQS) for heavy PDF indexing.

## Troubleshooting

If backend startup shows `ValidationError: Settings` with missing env vars, create `.env` first:

```bash
cp .env.example .env
```

If you start uvicorn from `backend/`, this project also reads `../.env` automatically.

