variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "ghcr_image_url" {
  description = "The GHCR URL for the FastAPI Docker image"
  type        = string
  default     = "ghcr.io/gughans/llm_rag"
}

variable "image_tag" {
  description = "The tag for the Docker image (usually the git commit SHA)"
  type        = string
  default     = "latest"
}

variable "api_key_secret" {
  description = "The secret API key to secure the endpoints"
  type        = string
  sensitive   = true
  default     = "dummy_key_do_not_use_in_prod"
}
