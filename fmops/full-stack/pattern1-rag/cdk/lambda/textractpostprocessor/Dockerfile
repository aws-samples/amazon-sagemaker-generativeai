FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9
RUN /var/lang/bin/python -m pip install --upgrade pip
RUN python -m pip install schadem-tidp-manifest==0.0.8 marshmallow boto3 pandas fsspec s3fs --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app/* ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "textract_postprocessor.lambda_handler" ]
