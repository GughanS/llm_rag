output "alb_hostname" {
  description = "The DNS name of the Application Load Balancer to access the API."
  value       = aws_lb.main.dns_name
}
