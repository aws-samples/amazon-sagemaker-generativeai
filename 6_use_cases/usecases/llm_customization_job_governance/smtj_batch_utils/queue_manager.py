import json
import time
from botocore.exceptions import ClientError
from config import Resource, Resources


class TrainingQueueManager:
    TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}

    def __init__(self, batch_client, verbose=True):
        self._batch_client = batch_client
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # Service Environments
    def create_service_env(self, create_se_request):
        try:
            return self._batch_client.create_service_environment(**create_se_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                desc_resp = self._batch_client.describe_service_environments(
                    serviceEnvironments=[create_se_request["serviceEnvironmentName"]]
                )
                se = desc_resp["serviceEnvironments"][0]
                self._log(
                    f"Service environment already exists: {se['serviceEnvironmentArn']}"
                )
                return {
                    "serviceEnvironmentName": se["serviceEnvironmentName"],
                    "serviceEnvironmentArn": se["serviceEnvironmentArn"],
                }
            print(f"ERROR: {json.dumps(error.response, indent=4)}")
            raise error

    def wait_for_se_update(self, se_name, expected_status, expected_state="ENABLED"):
        while True:
            resp = self._batch_client.describe_service_environments(
                serviceEnvironments=[se_name]
            )
            if resp["serviceEnvironments"]:
                se = resp["serviceEnvironments"][0]
                if se["status"] == expected_status and se["state"] == expected_state:
                    break
                if se["status"] == "INVALID":
                    raise ValueError(
                        f"Something went wrong! {json.dumps(se, indent=4)}"
                    )
            elif expected_status == "DELETED":
                self._log(f"SE {se_name} has been deleted")
                break
            time.sleep(5)

    def delete_service_env(self, se_name):
        self._log(f"Setting SE {se_name} to DISABLED")
        self._batch_client.update_service_environment(
            serviceEnvironment=se_name, state="DISABLED"
        )
        self._log("Waiting for SE update to finish...")
        self.wait_for_se_update(se_name, "VALID", "DISABLED")

        self._log(f"Deleting SE {se_name}")
        self._batch_client.delete_service_environment(serviceEnvironment=se_name)
        self._log("Waiting for SE update to finish...")
        self.wait_for_se_update(se_name, "DELETED", "DISABLED")

    # Scheduling Policies
    def create_scheduling_policy(self, create_sp_request):
        try:
            return self._batch_client.create_scheduling_policy(**create_sp_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                sp_arn = None
                sp_name = None
                list_resp = self._batch_client.list_scheduling_policies()
                for sp in list_resp["schedulingPolicies"]:
                    name_from_arn = sp["arn"].split("scheduling-policy/")[-1]
                    if name_from_arn == create_sp_request["name"]:
                        sp_arn = sp["arn"]
                        sp_name = name_from_arn
                self._log(f"Scheduling Policy already exists: {sp_arn}")
                return {"name": sp_name, "arn": sp_arn}
            print(f"ERROR: {json.dumps(error.response, indent=4)}")
            raise error

    def delete_scheduling_policy(self, sp_arn):
        self._batch_client.delete_scheduling_policy(arn=sp_arn)

    # Job Queues
    def create_job_queue(self, create_jq_request):
        try:
            return self._batch_client.create_job_queue(**create_jq_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                desc_resp = self._batch_client.describe_job_queues(
                    jobQueues=[create_jq_request["jobQueueName"]]
                )
                jq = desc_resp["jobQueues"][0]
                self._log(f"Job queue already exists: {jq['jobQueueArn']}")
                return {
                    "jobQueueName": jq["jobQueueName"],
                    "jobQueueArn": jq["jobQueueArn"],
                }
            print(f"ERROR: {json.dumps(error.response, indent=4)}")
            raise error

    def wait_for_jq_update(self, jq_name, expected_status, expected_state="ENABLED"):
        while True:
            resp = self._batch_client.describe_job_queues(jobQueues=[jq_name])
            if resp["jobQueues"]:
                jq = resp["jobQueues"][0]
                if jq["status"] == expected_status and jq["state"] == expected_state:
                    break
                if jq["status"] == "INVALID":
                    raise ValueError(
                        f"Something went wrong! {json.dumps(jq, indent=4)}"
                    )
            elif expected_status == "DELETED":
                self._log(f"JQ {jq_name} has been deleted")
                break
            time.sleep(5)

    def delete_job_queue(self, jq_name):
        self._log(f"Disabling JQ {jq_name}")
        self._batch_client.update_job_queue(jobQueue=jq_name, state="DISABLED")
        self._log("Waiting for JQ update to finish...")
        self.wait_for_jq_update(jq_name, "VALID", "DISABLED")

        self._log(f"Deleting JQ {jq_name}")
        self._batch_client.delete_job_queue(jobQueue=jq_name)
        self._log("Waiting for JQ update to finish...")
        self.wait_for_jq_update(jq_name, "DELETED", "DISABLED")

    # Quota Shares
    def create_quota_share(self, create_qs_request):
        try:
            return self._batch_client.create_quota_share(**create_qs_request)
        except ClientError as error:
            if "already exists" in error.response["message"]:
                qs_name = create_qs_request["quotaShareName"]
                jq_name = create_qs_request["jobQueue"]
                desc_resp = self._batch_client.describe_job_queues(jobQueues=[jq_name])
                jq_arn = desc_resp["jobQueues"][0]["jobQueueArn"]
                qs_arn = f"{jq_arn}/quota-share/{qs_name}"
                self._log(f"Quota share already exists: {qs_arn}")
                return {
                    "quotaShareName": qs_name,
                    "quotaShareArn": qs_arn,
                }
            print(f"ERROR: {json.dumps(error.response, indent=4)}")
            raise error

    def wait_for_qs_update(self, qs_arn, expected_status, expected_state="ENABLED"):
        while True:
            try:
                resp = self._batch_client.describe_quota_share(quotaShareArn=qs_arn)
                if (
                    resp["status"] == expected_status
                    and resp["state"] == expected_state
                ):
                    break
                if resp["status"] == "INVALID":
                    raise ValueError(
                        f"Something went wrong! {json.dumps(resp, indent=4)}"
                    )
            except ClientError:
                if expected_status == "DELETED":
                    self._log(f"Quota share {qs_arn} has been deleted")
                    break
                raise
            time.sleep(5)

    def delete_quota_share(self, qs_arn):
        self._log(f"Disabling quota share {qs_arn}")
        self._batch_client.update_quota_share(quotaShareArn=qs_arn, state="DISABLED")
        self._log("Waiting for quota share update to finish...")
        self.wait_for_qs_update(qs_arn, "VALID", "DISABLED")

        self._log(f"Deleting quota share {qs_arn}")
        self._batch_client.delete_quota_share(quotaShareArn=qs_arn)
        self._log("Waiting for quota share deletion to finish...")
        self.wait_for_qs_update(qs_arn, "DELETED", "DISABLED")


def create_resources(queue_configs, manager):
    resources = Resources()

    # Deduplicate service environments (multiple queues could share one)
    seen_se = set()
    for qc in queue_configs:
        if qc.service_environment.name not in seen_se:
            seen_se.add(qc.service_environment.name)
            manager._log(f"Creating service environment: {qc.service_environment.name}")
            resp = manager.create_service_env(
                qc.service_environment.to_create_request()
            )
            manager.wait_for_se_update(resp["serviceEnvironmentName"], "VALID")
            resources.service_environments.append(
                Resource(
                    name=resp["serviceEnvironmentName"],
                    arn=resp["serviceEnvironmentArn"],
                )
            )

    # Create scheduling policies (fair-share and quota-management)
    seen_sp = set()
    sp_arn_map = {}
    for qc in queue_configs:
        policy = qc.scheduling_policy or qc.qm_policy
        if policy and policy.name not in seen_sp:
            seen_sp.add(policy.name)
            manager._log(f"Creating scheduling policy: {policy.name}")
            resp = manager.create_scheduling_policy(policy.to_create_request())
            resources.scheduling_policies.append(
                Resource(name=resp["name"], arn=resp["arn"])
            )
            sp_arn_map[resp["name"]] = resp["arn"]

    # Create job queues
    for qc in queue_configs:
        policy = qc.scheduling_policy or qc.qm_policy
        sp_arn = sp_arn_map.get(policy.name) if policy else None
        manager._log(f"Creating training job queue: {qc.name}")
        resp = manager.create_job_queue(
            qc.to_create_request(scheduling_policy_arn=sp_arn)
        )
        manager.wait_for_jq_update(resp["jobQueueName"], "VALID")
        resources.job_queues.append(
            Resource(name=resp["jobQueueName"], arn=resp["jobQueueArn"])
        )

    # Create quota shares (only for quota-management queues)
    for qc in queue_configs:
        if qc.quota_shares:
            for qs in qc.quota_shares:
                manager._log(f"Creating quota share: {qs.name} (queue: {qc.name})")
                resp = manager.create_quota_share(
                    qs.to_create_request(job_queue_name=qc.name)
                )
                manager.wait_for_qs_update(resp["quotaShareArn"], "VALID")
                resources.quota_shares.append(
                    Resource(name=resp["quotaShareName"], arn=resp["quotaShareArn"])
                )

    return resources


def delete_resources(resources, manager):
    total = (
        len(resources.quota_shares)
        + len(resources.job_queues)
        + len(resources.service_environments)
        + len(resources.scheduling_policies)
    )
    step = 0

    print(f"\nDeleting {total} resources...\n")

    # Delete quota shares first (before their parent queues)
    for qs in resources.quota_shares:
        step += 1
        print(f"[{step}/{total}] Deleting quota share: {qs.name}")
        manager.delete_quota_share(qs.arn)
        print(f"  -> Quota share '{qs.name}' deleted.\n")

    for queue in resources.job_queues:
        step += 1
        print(f"[{step}/{total}] Deleting job queue: {queue.name}")
        manager.delete_job_queue(queue.name)
        print(f"  -> Job queue '{queue.name}' deleted.\n")

    for se in resources.service_environments:
        step += 1
        print(f"[{step}/{total}] Deleting service environment: {se.name}")
        manager.delete_service_env(se.name)
        print(f"  -> Service environment '{se.name}' deleted.\n")

    for sp in resources.scheduling_policies:
        step += 1
        print(f"[{step}/{total}] Deleting scheduling policy: {sp.name}")
        manager.delete_scheduling_policy(sp.arn)
        print(f"  -> Scheduling policy '{sp.name}' deleted.\n")

    print("All resources removed.")
