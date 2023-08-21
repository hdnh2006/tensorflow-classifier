# Usage: docker build -t tensorflow-classifier .
# Usage: docker run --gpus all -it -e WANDB_API_KEY=your_api_key tensorflow-classifier


# Use an official TensorFlow runtime as a parent image with GPU support
FROM tensorflow/tensorflow:2.10.0-gpu
# FROM tensorflow/tensorflow:2.13.0-gpu


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update and install some libraries
RUN apt-get update && apt install ffmpeg libsm6 libxext6 -y # Warning: It seems some endpoints from NVIDIA are currently broken, run after container initializes.

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Define environment variable
ENV NAME World

# Set default command to bash shell
CMD ["/bin/bash"]
