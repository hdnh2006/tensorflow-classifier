# Usage: docker build -t my_image .
# Usage: docker run --gpus all -it my_image

# Use an official TensorFlow runtime as a parent image with GPU support
FROM tensorflow/tensorflow:2.5.0-gpu
# FROM tensorflow/tensorflow:2.13.0-gpu


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Define environment variable
ENV NAME World

# Set default command to bash shell
CMD ["/bin/bash"]
