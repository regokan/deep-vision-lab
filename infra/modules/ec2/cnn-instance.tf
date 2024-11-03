resource "aws_key_pair" "key_pair" {
  key_name   = "ec2-cnn-instance"
  public_key = file("${path.module}/ec2-cnn-instance.pub")
}

resource "aws_instance" "cnn_instance" {
  ami             = "ami-0866a3c8686eaeeba" # Ubuntu 24.04 LTS
  instance_type   = "p3.2xlarge"
  key_name        = aws_key_pair.key_pair.key_name
  security_groups = [var.aws_security_group_allow_ssh_name]

  # Specify root block device to set custom volume size
  root_block_device {
    volume_size = 50   # Set the volume size to 50GB
    volume_type = "gp3" # General Purpose SSD, "gp3" is more cost-effective
  }

  tags = {
    Name = "cnn-instance"
  }
}
