import ray
import re


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
    

def is_ray_node_resource(resource_key):
    """Check if the current resource is the host ip."""
    ishost_regex = re.compile(r"^node:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    return ishost_regex.match(resource_key)