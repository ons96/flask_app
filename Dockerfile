# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the requirements file first for better caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt using uv
# Use --system to install in the main environment within the container
RUN uv pip install --system -r requirements.txt

# Create a non-root user first
RUN useradd -m -u 1000 user

# Copy the rest of the application code into the container at /app
# Files will initially be owned by root
COPY . .

# Create the session directory and change ownership of the entire app directory
# This needs to run as root BEFORE switching user
RUN mkdir -p /app/flask_session && chown -R 1000:1000 /app

# Now switch to the non-root user
USER user

# Set path explicitly for the non-root user
ENV PATH="/home/user/.local/bin:${PATH}"

# Make port 7860 available (this doesn't actually publish the port, more for documentation)
EXPOSE 7860

# Define environment variable for the port
ENV PORT=7860

# Run app.py when the container launches using gunicorn
# Binding to 0.0.0.0 is essential inside the container
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "app:app"] 