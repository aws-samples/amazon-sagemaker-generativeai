### Create AWS Batch resources for SageMaker Training job

#### Define AWS Batch Queues

All queue configuration is defined in [`config.yaml`](./config.yaml). Each entry defines a queue along with its service environment, optional scheduling policy, and optional quota shares.

```yaml
queues:
  # Simple FIFO queue - jobs run in submission order
  - name: team-a-queue
    type: fifo
    priority: 2
    service_environment:
      name: team-a-service-environment
      max_capacity: 1

  # Fair-share queue - jobs are scheduled based on share identifier weights
  - name: team-b-queue
    type: fair-share
    priority: 1
    service_environment:
      name: team-b-service-environment
      max_capacity: 1
    scheduling_policy:
      name: team-b-policy
      share_decay_seconds: 0
      share_distribution:
        - identifier: HIGHPRI
          weight: 1
        - identifier: MIDPRI
          weight: 50
        - identifier: LOWPRI
          weight: 99

  # Quota management queue - capacity allocation with lending, borrowing, and preemption
  - name: team-c-queue
    type: quota-management
    priority: 1
    service_environment:
      name: team-c-service-environment
      capacity_limits:
        - capacity_unit: ml.g5.xlarge
          max_capacity: 3
    scheduling_policy:
      name: team-c-policy
    quota_shares:
      - name: QS1
        capacity_limits:
          - capacity_unit: ml.g5.xlarge
            max_capacity: 1
        sharing_strategy: LEND_AND_BORROW
        borrow_limit: 200
        in_share_preemption: true
      - name: QS2
        capacity_limits:
          - capacity_unit: ml.g5.xlarge
            max_capacity: 1
        sharing_strategy: LEND
        in_share_preemption: false
```

##### Configuration reference

**Queue fields**

| Field                                   | Required                           | Description                                                                                                                                |
| --------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`                                  | Yes                                | Name of the job queue                                                                                                                      |
| `type`                                  | Yes                                | `fifo`, `fair-share`, or `quota-management`                                                                                                |
| `priority`                              | Yes                                | Queue priority (higher number = higher priority). Matters when multiple queues share a service environment                                 |
| `service_environment.name`              | Yes                                | Name of the service environment                                                                                                            |
| `service_environment.max_capacity`      | Yes (fifo, fair-share)             | Maximum number of concurrent training instances                                                                                            |
| `service_environment.capacity_limits`   | Yes (quota-management)             | List of `{capacity_unit, max_capacity}` per instance type (max 5)                                                                          |
| `scheduling_policy.name`                | Yes (fair-share, quota-management) | Name of the scheduling policy                                                                                                              |
| `scheduling_policy.share_decay_seconds` | No                                 | Half-life for usage-based rebalancing. `0` (default) = pure weight-based priority. Higher values = past usage influences scheduling longer |
| `scheduling_policy.share_distribution`  | Yes (fair-share)                   | List of share identifiers with weights. Lower weight = more resources = higher priority                                                    |

**Quota share fields** (only for `quota-management` queues)

| Field                 | Required              | Description                                                                                                      |
| --------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `name`                | Yes                   | Name of the quota share                                                                                          |
| `capacity_limits`     | Yes                   | List of `{capacity_unit, max_capacity}` per instance type (1-5 entries)                                          |
| `sharing_strategy`    | Yes                   | `RESERVE`, `LEND`, or `LEND_AND_BORROW`                                                                          |
| `borrow_limit`        | Yes (LEND_AND_BORROW) | Maximum borrow percentage (e.g., `200` = can borrow up to 200% of own capacity)                                  |
| `in_share_preemption` | No                    | `true` or `false` (default). When enabled, higher-priority jobs can preempt lower-priority jobs within the share |

##### Share identifier constraints (fair-share)

- Only **alphanumeric characters** are allowed (`A-Z`, `a-z`, `0-9`), with an optional trailing `*`
- Maximum 255 characters
- No underscores, hyphens, or special characters
- Maximum 500 active share identifiers per queue

##### Queue types

1. **FIFO** - Jobs run in submission order. No scheduling policy needed.
2. **Fair-share** - Jobs are scheduled based on share identifier weights. Requires a `scheduling_policy` section. When submitting jobs, you must specify both a `share_identifier` and a `priority` (0-9999). The priority orders jobs within the same share identifier (higher number = runs first).
3. **Quota management** - Jobs are dispatched through quota shares, each with explicit capacity limits and resource sharing strategies. Supports lending idle capacity between shares, borrowing from other shares, and preemption (both cross-share and in-share). When submitting jobs, specify a `quota_share_name` and optionally a `priority`.

##### Resource sharing strategies (quota-management)

| Strategy          | Lends idle capacity | Borrows idle capacity | Notes                                                                              |
| ----------------- | ------------------- | --------------------- | ---------------------------------------------------------------------------------- |
| `RESERVE`         | No                  | No                    | Capacity is exclusively reserved                                                   |
| `LEND`            | Yes                 | No                    | Idle capacity available to other shares                                            |
| `LEND_AND_BORROW` | Yes                 | Yes                   | Requires `borrow_limit`. Cross-share preemption reclaims lent capacity when needed |

##### Preemption (quota-management)

- **Cross-share preemption** is always active. When a quota share needs capacity it previously lent out, AWS Batch preempts jobs that borrowed it.
- **In-share preemption** is opt-in per quota share. When enabled, higher-priority `RUNNABLE` jobs can preempt lower-priority `RUNNING`/`SCHEDULED`/`STARTING` jobs within the same share.
- Job priority can be **updated after submission** using `UpdateServiceJob` / `job.update(scheduling_priority=N)`.

#### Create resources

```bash
python3 create_resources.py
```

You can also specify a custom config file:

```bash
python3 create_resources.py --config /path/to/my-config.yaml
```

#### Delete resources

```bash
python3 create_resources.py --clean
```

This deletes all quota shares, queues, service environments, and scheduling policies defined in the config, showing progress for each step.
