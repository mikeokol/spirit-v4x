# Deployment Cheat-Sheet

## Fly.io (free tier)
```bash
flyctl apps create spirit-backend
flyctl secrets set JWT_SECRET=$(openssl rand -hex 32)
flyctl deploy --dockerfile docker/Dockerfile
