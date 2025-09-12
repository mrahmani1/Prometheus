# Required GitHub Secrets Configuration

This document outlines all the secrets that need to be configured in your GitHub repository for the CI/CD pipeline to work properly.

## Repository Secrets

Navigate to your repository → Settings → Secrets and variables → Actions to configure these secrets:

### Code Quality & Security
- `SONAR_TOKEN` - SonarQube authentication token for code quality analysis
- `SNYK_TOKEN` - Snyk authentication token for security vulnerability scanning

### Google Cloud Platform (GCP)
- `GCP_SA_KEY` - Service account key JSON (base64 encoded) for GCP authentication
- `GCP_PROJECT_ID` - Your GCP project ID

### Azure Cloud
- `AZURE_CLIENT_ID` - Azure service principal client ID
- `AZURE_CLIENT_SECRET` - Azure service principal client secret
- `AZURE_TENANT_ID` - Azure tenant ID

### Amazon Web Services (AWS)
- `AWS_ACCESS_KEY_ID` - AWS access key for programmatic access
- `AWS_SECRET_ACCESS_KEY` - AWS secret access key

## Environment-Specific Secrets

### Staging Environment
Configure these in GitHub → Settings → Environments → staging:
- Database connection strings
- API keys for external services
- SSL certificates

### Production Environment
Configure these in GitHub → Settings → Environments → production:
- Production database credentials
- Production API keys
- SSL certificates
- Monitoring service tokens

## How to Generate Secrets

### GCP Service Account Key
```bash
# Create service account
gcloud iam service-accounts create prometheus-ci-cd

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:prometheus-ci-cd@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"

# Create and download key
gcloud iam service-accounts keys create key.json \
  --iam-account=prometheus-ci-cd@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Base64 encode for GitHub secret
base64 -i key.json
```

### Azure Service Principal
```bash
# Create service principal
az ad sp create-for-rbac --name "prometheus-ci-cd" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID
```

### AWS IAM User
```bash
# Create IAM user with programmatic access
aws iam create-user --user-name prometheus-ci-cd

# Attach necessary policies
aws iam attach-user-policy --user-name prometheus-ci-cd \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

# Create access keys
aws iam create-access-key --user-name prometheus-ci-cd
```

## Security Best Practices

1. **Principle of Least Privilege** - Grant only the minimum permissions required
2. **Regular Rotation** - Rotate secrets regularly (every 90 days recommended)
3. **Environment Separation** - Use different credentials for staging and production
4. **Audit Logging** - Enable audit logging for all cloud services
5. **Secret Scanning** - Enable GitHub secret scanning to prevent accidental exposure