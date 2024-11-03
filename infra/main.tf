module "s3" {
  source = "./modules/ec2"

  aws_security_group_allow_ssh_name = aws_security_group.allow_ssh.name
}
