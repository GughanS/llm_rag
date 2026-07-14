# AWS Infrastructure as Code (IaC)

This directory contains the theoretical Terraform infrastructure to deploy the LLM serving API to AWS.

> **Note:** As specified in the Architecture Decision Record (ADR), this infrastructure is strictly **documented, not deployed**. This avoids incurring cloud costs for a portfolio project, while explicitly demonstrating knowledge of modern cloud architecture and HashiCorp Terraform.

## Architecture Diagram (Conceptual)

1. **Client** requests hit an **Application Load Balancer (ALB)** sitting in public subnets.
2. The ALB terminates TLS and forwards the traffic to an **ECS Fargate Cluster** in private subnets.
3. The Fargate tasks (running our FastAPI monolith Docker image from GHCR) handle the generation.
4. The rate limiter inside the API connects to an **ElastiCache Redis** cluster in the private subnets.

## Components Provisioned

- **VPC & Networking:** A secure VPC utilizing `terraform-aws-modules` with Public and Private subnets, plus a NAT Gateway.
- **ElastiCache Redis:** A `t4g.micro` Redis instance for the token-bucket rate limiter.
- **ECS Fargate:** Serverless container execution for the FastAPI monolith.
- **Application Load Balancer (ALB):** Routes HTTP/HTTPS traffic to the Fargate instances.
- **Security Groups:** Strictly limits traffic (ALB -> ECS -> Redis).

## How to Deploy (If desired)

1. Configure AWS credentials (`aws configure`).
2. Initialize Terraform:
   ```bash
   terraform init
   ```
3. Plan the deployment to see the resources that would be created:
   ```bash
   terraform plan -var="api_key_secret=YOUR_SECURE_KEY"
   ```
4. (Optional) Apply the infrastructure:
   ```bash
   terraform apply -var="api_key_secret=YOUR_SECURE_KEY"
   ```
