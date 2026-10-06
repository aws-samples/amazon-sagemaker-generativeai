"""Deploy and delete the SageMaker AI real-time endpoints that serve the models under evaluation.

    dep = EndpointDeployer(region, role, serving="vllm", instance_types=[...], max_model_len=40960)
    base = dep.deploy("base", "Qwen/Qwen3.5-4B")
    student = dep.deploy("student", "/opt/ml/model", dep.student_model_data(model_package_arn))
    dep.delete(base)

Two serving containers are supported; both expose the endpoint's OpenAI-compatible path and honor per-request
chat_template_kwargs (Qwen3.5's thinking switch):
  "vllm": SageMaker vLLM Deep Learning Container; SM_VLLM_* environment variables become vLLM server flags.
  "djl":  DJL/LMI container (vLLM engine inside); OPTION_* environment variables.
Native tool-call parsing stays off on purpose: ShoppingBench reads tool calls from the reply text.
"""
import time
import boto3

IMAGES = {
    "djl": "763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.36.0-lmi25.0.0-cu130",
    "vllm": "763104351884.dkr.ecr.{region}.amazonaws.com/vllm:0.20.2-gpu-py312-cu130-ubuntu22.04-sagemaker",
}
# Failures worth retrying on the next instance type rather than raising
RETRYABLE = ("InsufficientInstanceCapacity", "CannotStartContainerError")


class EndpointDeployer:
    def __init__(self, region, role, serving="vllm", instance_types=("ml.g6e.xlarge",), max_model_len=40960,
                 max_num_seqs=16, stamp=None):
        assert serving in IMAGES, f"serving must be one of {list(IMAGES)}"
        self.region, self.role, self.serving = region, role, serving
        self.instance_types, self.max_model_len, self.max_num_seqs = list(instance_types), max_model_len, max_num_seqs
        self.image = IMAGES[serving].format(region=region)
        self.stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.sm = boto3.client("sagemaker", region_name=region)

    # ---------- model location ----------
    def student_model_data(self, model_package_arn):
        """A Model Package registered by a serverless training job points at the job's whole output prefix
        (checkpoints/, global_step_*/, metrics), not at a flat Hugging Face folder. Return the ModelDataSource of the
        folder that has config.json + safetensors: the root, then checkpoints/hf_merged/ (merged LoRA), then the
        latest global_step_*/huggingface/."""
        c = self.sm.describe_model_package(ModelPackageName=model_package_arn)["InferenceSpecification"]["Containers"][0]
        if not c.get("ModelDataSource"):
            return {"ModelDataUrl": c["ModelDataUrl"]}
        root = c["ModelDataSource"]["S3DataSource"]["S3Uri"].rstrip("/") + "/"
        bucket, key = root[5:].split("/", 1)
        s3 = boto3.client("s3", region_name=self.region)
        keys = [o["Key"][len(key):] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key)
                for o in page.get("Contents", [])]
        steps = sorted({k.split("/")[0] for k in keys if k.startswith("global_step_")}, key=lambda d: int(d.split("_")[-1]))
        for sub in ["", "checkpoints/hf_merged/"] + [f"{d}/huggingface/" for d in reversed(steps)]:
            if f"{sub}config.json" in keys and any(k.startswith(sub) and k.endswith(".safetensors")
                                                    and "/" not in k[len(sub):] for k in keys):
                print("student weights:", root + sub)
                return {"ModelDataSource": {"S3DataSource": {"S3Uri": root + sub, "S3DataType": "S3Prefix",
                                                             "CompressionType": "None"}}}
        raise RuntimeError(f"no merged Hugging Face folder (config.json + *.safetensors) under {root}")

    def container_env(self, hf_model_id):
        """hf_model_id: a Hugging Face Hub id (downloaded at startup) or /opt/ml/model (the attached S3 weights)."""
        if self.serving == "djl":
            return {"HF_MODEL_ID": hf_model_id, "OPTION_ROLLING_BATCH": "vllm", "OPTION_TRUST_REMOTE_CODE": "true",
                    "OPTION_TENSOR_PARALLEL_DEGREE": "1", "OPTION_MAX_MODEL_LEN": str(self.max_model_len),
                    "OPTION_MAX_ROLLING_BATCH_SIZE": str(self.max_num_seqs)}
        return {"HF_MODEL_ID": hf_model_id, "SM_VLLM_MAX_MODEL_LEN": str(self.max_model_len),
                "SM_VLLM_TRUST_REMOTE_CODE": "true", "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
                "SM_VLLM_MAX_NUM_SEQS": str(self.max_num_seqs), "SAGEMAKER_ENABLE_LOAD_AWARE": "1"}

    # ---------- lifecycle ----------
    def status(self, name):
        """Endpoint status, or None if no endpoint of this name exists. Other errors (e.g. expired credentials)
        are raised rather than reported as a missing endpoint."""
        try:
            return self.sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
        except self.sm.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                return None
            raise

    def delete(self, name, wait=True):
        """Delete the endpoint, endpoint config and model of this name (missing pieces are skipped)."""
        for fn, kw in ((self.sm.delete_endpoint, "EndpointName"), (self.sm.delete_endpoint_config, "EndpointConfigName"),
                       (self.sm.delete_model, "ModelName")):
            try:
                fn(**{kw: name})
            except Exception:
                pass
        while wait and self.status(name):
            time.sleep(15)

    def _create(self, name, itype, container):
        self.sm.create_model(ModelName=name, ExecutionRoleArn=self.role, PrimaryContainer=container)
        self.sm.create_endpoint_config(EndpointConfigName=name, ProductionVariants=[{
            "VariantName": "variant1", "ModelName": name, "InstanceType": itype, "InitialInstanceCount": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800}])
        self.sm.create_endpoint(EndpointName=name, EndpointConfigName=name)

    def _wait(self, name):
        while (d := self.sm.describe_endpoint(EndpointName=name))["EndpointStatus"] in ("Creating", "Updating"):
            time.sleep(30)
        return d

    def _name(self, tag, itype):        # e.g. seqkd-eval-base-g6e-xlarge-20261005-120000
        return f"seqkd-eval-{tag}-{itype.split('.', 1)[1].replace('.', '-')}-{self.stamp}"

    def deploy(self, tag, hf_model_id, model_data=None):
        """Create an endpoint on the first instance type that has capacity and starts the container. Idempotent:
        re-running reuses an endpoint of the same name that is Creating/InService. Returns the name once InService."""
        for itype in self.instance_types:
            name = self._name(tag, itype)
            if self.status(name) in ("Creating", "Updating", "InService"):
                print(f"{tag}: reusing endpoint {name}")
            else:
                self.delete(name)       # leftovers from an interrupted run (model/config without endpoint, or Failed)
                self._create(name, itype, {"Image": self.image, "Environment": self.container_env(hf_model_id),
                                           **(model_data or {})})
            d = self._wait(name)
            if d["EndpointStatus"] == "InService":
                print(f"{tag}: InService on {itype} -> {name}")
                return name
            reason = d.get("FailureReason", "")
            self.delete(name)
            if not any(r in reason for r in RETRYABLE):
                raise RuntimeError(f"{tag} endpoint failed on {itype}: {reason}")
            print(f"{tag}: {itype} failed ({reason.split('.')[0]}), trying the next instance type")
        raise RuntimeError(f"no capacity for any of {self.instance_types}; retry later or add types")

    def deploy_student_with_modelbuilder(self, model_package_arn, itype="ml.p4d.24xlarge"):
        """Alternative student deployment through the SageMaker Python SDK's ModelBuilder (DJL/LMI only).

        ModelBuilder resolves the Model Package (container image + checkpoints/hf_merged/) and applies the base model's
        hosting recipe. The recipe environment overrides any env_vars you pass (it is published for 4-GPU g6/g6e:
        tensor parallel 4, eager mode), so this keeps what ModelBuilder resolved and deploys it with boto3 using one
        GPU, CUDA graphs on, and the long context."""
        assert self.serving == "djl", "ModelBuilder deploys the DJL/LMI container: use serving='djl'"
        from sagemaker.serve import ModelBuilder, ModelServer
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        from sagemaker.core.resources import ModelPackage
        schema = SchemaBuilder(sample_input={"inputs": "Hello", "parameters": {"max_new_tokens": 16}},  # required for
                               sample_output=[{"generated_text": "Hi"}])                                # DJL_SERVING
        mb = ModelBuilder(model=ModelPackage.get(model_package_name=model_package_arn), role_arn=self.role,
                          image_uri=self.image, model_server=ModelServer.DJL_SERVING, schema_builder=schema,
                          instance_type=itype)
        built = mb.build()                               # creates a SageMaker Model with the recipe env
        bname = getattr(built, "model_name", None) or mb.model_name
        d = self.sm.describe_model(ModelName=bname)
        pc = d.get("PrimaryContainer") or d["Containers"][0]
        print("ModelBuilder resolved:", pc["Image"], pc["ModelDataSource"]["S3DataSource"]["S3Uri"])
        env = {**pc["Environment"], "OPTION_TENSOR_PARALLEL_DEGREE": "1", "OPTION_ENFORCE_EAGER": "false",
               "OPTION_MAX_MODEL_LEN": str(self.max_model_len)}
        name = self._name("student-mb", itype)
        if self.status(name) not in ("Creating", "Updating", "InService"):
            self.delete(name)
            self._create(name, itype, {"Image": pc["Image"], "ModelDataSource": pc["ModelDataSource"], "Environment": env})
        self.sm.delete_model(ModelName=bname)            # the recipe-env model is not used
        d = self._wait(name)
        if d["EndpointStatus"] != "InService":
            raise RuntimeError(f"student (ModelBuilder) failed on {itype}: {d.get('FailureReason')}")
        print("student (ModelBuilder): InService ->", name)
        return name
