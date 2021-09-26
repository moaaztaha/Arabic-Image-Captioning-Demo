# Set base image (host OS)
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

RUN apt update
RUN apt install python3 python3-pip -y
RUN pip3 install --upgrade pip setuptools wheel

# Copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# Install any dependencies
RUN pip3 install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD [ "python3", "./application.py" ]