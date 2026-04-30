from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_worker_count(value: object, *, default: int = 0) -> int:
    if value in {None, "", "none"}:
        return int(default)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return auto_worker_count()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(parsed, 0)


def auto_worker_count(*, reserve_cores: int = 1, minimum: int = 1) -> int:
    cpu_count = os.cpu_count() or 1
    return max(cpu_count - int(reserve_cores), int(minimum))


def iter_parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
    use_process_pool: bool,
    in_flight_multiplier: int = 2,
    start_method: str = "spawn",
) -> Iterator[R]:
    if max_workers <= 1:
        for item in items:
            yield fn(item)
        return

    in_flight_limit = max(int(max_workers) * max(int(in_flight_multiplier), 1), 1)
    iterator = iter(items)
    futures: deque = deque()

    if use_process_pool:
        mp_context = mp.get_context(start_method)
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {"mp_context": mp_context}
    else:
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {}

    with executor_cls(max_workers=max_workers, **executor_kwargs) as executor:
        for _ in range(in_flight_limit):
            item = next(iterator, None)
            if item is None:
                break
            futures.append(executor.submit(fn, item))
        while futures:
            future = futures.popleft()
            yield future.result()
            item = next(iterator, None)
            if item is not None:
                futures.append(executor.submit(fn, item))
