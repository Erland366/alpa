import logging
import ray
from ray.util.placement_group import get_current_placement_group,\
    PlacementGroup
import time
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def try_import_ray_worker(error: bool = False):
    """Tries importing `ray.worker` and returns the module (or None).

    Args:
        error: Whether to raise an error if ray.worker cannot be imported.

    Returns:
        The `ray.worker` modules.

    Raises:
        ImportError: If error=True and ray's version >= 2.0.
    """
    # In the ray-nightly version,
    # worker = _DeprecationWrapper("worker", ray._private.worker)
    # `_DeprecationWrapper` has attributes of `_real_worker`
    try:
        if hasattr(ray.worker, "_real_worker"):
            if error:
                raise ImportError("Could not import `ray.worker`!"
                                  "You might use the ray-nightly "
                                  "and `ray.worker` is deprecated there"
                                  "`pip install ray==1.13.0`.")
            return ray.worker._real_worker  # pylint: disable=protected-access
        else:
            return ray.worker
    except ModuleNotFoundError:
        return ray._private.worker  # pylint: disable=protected-access
    

def create_placement_group(num_hosts,
                           host_num_devices,
                           name,
                           additional_resources_per_host=None):
    """Creates a placement group if it does not exist.

    If a placement group is already detected (in Tune integration),
    this will be a no-op.

    By default the placement group will be created with `SPREAD` strategy.
    This is optimized for colocating GPUs on different nodes.

    Args:
        num_hosts: the number of hosts to create the placement group for
        host_num_devices: the number of devices on each host
        additional_resources_per_host: additional resources per host

    Returns:
        The placement group
    """
    current_placement_group = get_current_placement_group()
    ray_worker = try_import_ray_worker()
    worker = ray_worker.global_worker  # pylint: disable=protected-access
    should_capture_child_tasks_in_placement_group = (
        worker.should_capture_child_tasks_in_placement_group)
    should_create_placement_group = (
        current_placement_group is None or
        not should_capture_child_tasks_in_placement_group)

    if should_create_placement_group:
        # `should_create_placement_group` is always True when using alpa alone.
        # `should_create_placement_group` can be false when integrated with Tune
        additional_resources_per_host = (additional_resources_per_host or {})
        bundles = [{
            "CPU": 1,
            "GPU": host_num_devices[i],
            **additional_resources_per_host
        } for i in range(num_hosts)]

        # Alpa Placement Group: `SPREAD` strategy is required
        # https://docs.ray.io/en/latest/ray-core/placement-group.html#strategy-types
        # Each bundle must be scheduled in a separate node.
        strategy = "SPREAD"

        placement_group = ray.util.placement_group(bundles,
                                                   strategy=strategy,
                                                   name=name or "")
        logger.info("Waiting for placement group to start.")
        timeout = 100
        ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
        if ready:
            logger.info("Placement group has started.")
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure your "
                "cluster either has enough resources or use an "
                "autoscaling cluster. If you are running on a cluster, "
                "make sure you specify an address in `ray.init()`, for example,"
                ' `ray.init("auto")`. You can also increase the timeout by '
                "setting the ALPA_PLACEMENT_GROUP_TIMEOUT_S environment "
                "variable. Current resources available: "
                f"{ray.available_resources()}, resources requested by "
                f"the placement group: {placement_group.bundle_specs}")
        return placement_group
    else:
        return current_placement_group
    

if __name__=="__main__":
    pg = create_placement_group(num_hosts=1, host_num_devices=[2], name="alpa")
    time.sleep(100)