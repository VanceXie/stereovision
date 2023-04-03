# -*- coding: UTF-8 -*-
import logging
import time


def get_var_name(var):
    return (name := [k for k, v in globals().items() if v is var]) and name[0] or None


# 装饰器函数，用于评估被装饰函数的运行时间
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Function {} took {:.6f} seconds to execute".format(func.__name__, end_time - start_time))
        return result
    
    return wrapper


# 装饰器函数用于输出log
def log(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Function {func.__name__} started with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Function {func.__name__} finished with result: {result}")
        return result
    
    return wrapper
