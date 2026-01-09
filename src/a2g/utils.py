import logging
import inspect
from functools import wraps
from typing import Iterable, Callable, TypeVar, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


def default_to_self(params: Iterable[str], *, verbose: bool = True) -> Callable[[T], T]:
    """
    Decorator to automatically default certain function arguments to the
    corresponding attributes of ``self`` if the argument is ``None``.

    Only arguments listed in ``params`` are affected.

    Behavior:
        - If the argument is ``None`` AND ``self`` has an attribute with the same name,
          the argument will be replaced by ``getattr(self, name)``.
        - Otherwise, the original argument value is retained.

    Parameters
    ----------
    params : Iterable[str]
        Names of parameters to default from ``self``. Must match attribute names.
    verbose : bool, optional
        If ``True``, logs when a parameter is ``None`` but no corresponding self attribute exists, by default ``True``.

    Returns
    -------
    decorator : Callable
        Function decorator.
    """
    params_set = set(params)

    def decorator(func: T) -> T:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Bind positional and keyword arguments
            bound = sig.bind_partial(self, *args, **kwargs)
            bound.apply_defaults()

            # Replace None arguments with self attributes when available
            for name in params_set:
                if name not in bound.arguments:
                    continue

                if bound.arguments[name] is None:
                    if hasattr(self, name):
                        bound.arguments[name] = getattr(self, name)
                    else:
                        if verbose:
                            logger.info(
                                "%s.%s: '%s' is None and not found on self; keep None.",
                                self.__class__.__name__,
                                func.__name__,
                                name,
                            )

            # Call the original function with updated arguments
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
