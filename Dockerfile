# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# Be Aware! For the Jenkins CI/CD pipeline, 
# input args are defined inside the JenkinsConstants.groovy, not here!

ARG tag=latest

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM ai4oshub/ai4os-yolo-torch:${tag}

LABEL maintainer='Jeremy Fix, Martin Laviale'
LABEL version='0.0.3'
# Diatom classification 

# Download new model weights and remove old ones
# You can use the following as "reference" - https://github.com/ai4os-hub/ai4os-image-classification-tf/blob/master/Dockerfile
###############
### FILL ME ###
###############

# Define default YoloV8 models
ENV YOLO_DEFAULT_WEIGHTS="yolov8_diamorph_medium"
ENV YOLO_DEFAULT_TASK_TYPE="cls"

# Uninstall existing module ("ai4os_yolo")
# Update MODEL_NAME to cold_coral_segmentation
# Copy updated pyproject.toml to include cold_coral_segmentation authors and rename the module
# Re-install application with the updated pyproject.toml
RUN cd /srv/ai4os-yolo-torch && \
    module=$(cat pyproject.toml |grep '\[project\]' -A1 |grep 'name' | cut -d'=' -f2 |tr -d ' ' |tr -d '"') && \
    pip uninstall -y $module
ENV MODEL_NAME="diamorph_classification"
COPY ./pyproject-child.toml /srv/ai4os-yolo-torch/pyproject.toml
RUN cd /srv/ai4os-yolo-torch && pip install --no-cache -e .

RUN mkdir -p /srv/ai4os-yolo-torch/models/$YOLO_DEFAULT_WEIGHTS/weights && \
    curl -L	https://github.com/ai4os-hub/diamorph-classification/releases/download/v2/species_yolov8l_best.pt \
    --output /srv/ai4os-yolo-torch/models/$YOLO_DEFAULT_WEIGHTS/weights/best.pt

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
  CMD curl --fail http://localhost:5000/v2  
