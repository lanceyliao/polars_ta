"""
这是对 helper.py 中命名空间方案的再封装，将成员函数调用转换为前缀函数调用

设计目的：
1. 将 expr.ta.func(...) 的链式调用转换为 func(expr, ...) 的前缀调用
2. 方便遗传算法等工具使用（遗传算法需要字符串形式的表达式）
3. 保持与 helper.py 相同的功能和参数

使用方式对比：

之前（helper.py）：
expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

现在（wrapper.py）：
func(expr, ..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

示例对比：
# helper.py 方式
pl.col('A').ta.BBANDS(timeperiod=20)

# wrapper.py 方式
BBANDS(pl.col('A'), timeperiod=20)

优点：
1. 前缀表达式更符合遗传算法的输入格式（字符串表达式）
2. 可以直接作为函数调用，不依赖 polars 的命名空间注册
3. 更容易进行代码生成和表达式解析

缺点：
1. 失去了 IDE 的智能提示（因为函数是动态生成的）
2. 需要调用 init() 来初始化环境
"""

from functools import wraps

import talib as _talib
from polars import Expr
from polars import struct
from talib import abstract as _abstract

from helper import TaLibHelper
# from polars_ta.utils.helper import TaLibHelper

_ = TaLibHelper


def ta_func(func, func_name, input_names, output_names,
            *args,
            skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
            **kwargs):
    """
    TA-Lib 函数包装器，将前缀调用转换为命名空间调用

    核心作用：
    1. 接收前缀函数调用的参数，如 BBANDS(pl.col('A'), timeperiod=20)
    2. 分离出 polars 表达式和其他参数
    3. 将表达式转换为命名空间调用，如 pl.col('A').ta.BBANDS(timeperiod=20)
    4. 调用 helper.py 中的 TaLibHelper 实际执行

    Parameters
    ----------
    func : callable
        原始的 TA-Lib 函数，如 talib.BBANDS
    func_name : str
        函数名称，如 'BBANDS'
    input_names : list of str
        TA-Lib 函数的输入参数名称列表（从函数元数据中获取）
    output_names : list of str
        TA-Lib 函数的输出参数名称列表（从函数元数据中获取）
    *args : tuple
        混合参数，包含 polars 表达式和其他参数
        例如：(pl.col('A'), 20) 或 (pl.col('A'), pl.col('B'), pl.col('C'), 2)
    skip_nan, output_idx, schema, schema_format, nan_to_null
        传递给 helper.py 的参数
    **kwargs : dict
        命名参数，如 timeperiod=20

    Returns
    -------
    Expr
        返回 polars 表达式

    """
    # 分离参数：将 polars 表达式和其他参数分开
    exprs = [arg for arg in args if isinstance(arg, Expr)]  # 提取所有 polars 表达式
    param = [arg for arg in args if not isinstance(arg, Expr)]  # 提取其他参数（数字、字符串等）

    # 根据表达式数量决定调用方式
    if len(exprs) == 1:
        # 单列输入：直接使用 expr.ta.func_name
        # 例如：pl.col('A').ta.BBANDS
        ef = getattr(exprs[0].ta, func_name)
    else:
        # 多列输入：先组合成 struct，再调用 ta.func_name
        # 例如：pl.struct(['A', 'B', 'C']).ta.ATR
        ef = getattr(struct(*exprs).ta, func_name)

    # 调用命名空间方法，传入参数
    return ef(*param,
              skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
              **kwargs)


def ta_decorator(func, func_name, input_names, output_names):
    """
    装饰器工厂，为 TA-Lib 函数创建包装函数

    核心作用：
    1. 接收原始 TA-Lib 函数及其元数据
    2. 返回一个装饰后的函数，该函数会调用 ta_func
    3. 保持原始函数的元信息（通过 @wraps）

    Parameters
    ----------
    func : callable
        原始的 TA-Lib 函数，如 talib.BBANDS
    func_name : str
        函数名称，如 'BBANDS'
    input_names : list of str
        TA-Lib 函数的输入参数名称列表
    output_names : list of str
        TA-Lib 函数的输出参数名称列表

    Returns
    -------
    function
        装饰后的函数，可以直接调用

    """
    @wraps(func)  # 保留原始函数的元信息（如 __name__, __doc__）
    def decorated(*args,
                  skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                  **kwargs):
        # 调用 ta_func 实际执行转换和调用
        return ta_func(func, func_name, input_names, output_names,
                       *args,
                       skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
                       **kwargs)

    return decorated


def init(to_globals=False, name_format='{}'):
    """
    初始化环境，动态生成所有 TA-Lib 函数的包装器

    核心作用：
    1. 遍历 TA-Lib 中的所有函数
    2. 为每个函数创建装饰器包装
    3. 将包装后的函数注册到全局变量或返回对象中

    Parameters
    ----------
    to_globals : bool
        是否注册到全局变量中
        - True: 函数会注册到调用者的全局命名空间，可以直接使用，如 BBANDS(...)
        - False: 函数只注册到返回的对象中，需要通过对象调用，如 t.BBANDS(...)
    name_format : str
        函数名格式字符串，用于为函数名添加前缀
        - '{}': 保持原名，如 'BBANDS'
        - 'ts_{}': 添加前缀，如 'ts_BBANDS'
        - 'XX_{}_YY': 添加前后缀，如 'XX_BBANDS_YY'

    Returns
    -------
    TA_LIB
        包含所有 TA-Lib 包装函数的对象实例
        如果 to_globals=False，需要通过这个对象调用函数

    Examples
    --------
    # 方式1：注册到全局变量
    init(to_globals=True, name_format='{}')
    BBANDS(pl.col('A'), timeperiod=20)  # 直接调用

    # 方式2：不注册到全局变量
    t = init(to_globals=False, name_format='ts_{}')
    t.ts_BBANDS(pl.col('A'), timeperiod=20)  # 通过对象调用

    """
    # 创建一个空类作为容器
    class TA_LIB:
        pass

    lib = TA_LIB()

    # 遍历 TA-Lib 中的所有函数
    for i, func_name in enumerate(_talib.get_functions()):
        """talib遍历"""
        # 获取原始 TA-Lib 函数
        _ta_func = getattr(_talib, func_name)

        # 获取函数的元数据（输入输出参数名称）
        info = _abstract.Function(func_name).info
        output_names = info['output_names']
        input_names = info['input_names']

        # 创建装饰后的包装函数
        f = ta_decorator(_ta_func, func_name, input_names, output_names)

        # 应用名称格式化
        name = name_format.format(func_name)

        # 将函数添加到对象中
        setattr(lib, name, f)

        # 如果需要注册到全局变量
        if to_globals:
            from inspect import currentframe
            # 获取调用者的栈帧（当前函数的上一帧）
            frame = currentframe().f_back
            # 将函数注册到调用者的全局命名空间
            frame.f_globals[name] = f

    return lib
