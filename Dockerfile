# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt using uv
# Use --system to install in the main environment within the container
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 7860 available to the world outside this container (standard for HF Spaces)
EXPOSE 7860

# Define environment variable for the port (optional, but good practice)
ENV PORT=7860

# Run app.py when the container launches using gunicorn
# Bind to 0.0.0.0 to accept connections from outside the container
# Use a single worker for simplicity on free tiers
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "app:app"] 