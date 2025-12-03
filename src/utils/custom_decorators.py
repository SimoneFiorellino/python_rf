import time


def timer(
    f,
    name=None,
):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        if name:
            print(f"Function '{name}' executed in {end_time - start_time:.4f} seconds")
        else:
            print(
                f"Function '{f.__name__}' executed in {end_time - start_time:.4f} seconds"
            )
        return result

    return wrapper
