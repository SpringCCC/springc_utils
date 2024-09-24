import time
from functools import wraps

def sc_timing_consume(logging=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            message = f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
            if logging is None:
                print(message)
            else:
                logging.info(message)
            return result
        return wrapper
    return decorator


class Line_ConsumeTime:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"代码执行时间: {self.execution_time:.6f} 秒")