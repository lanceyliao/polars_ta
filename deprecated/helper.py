"""
这个模块实现了 polars 表达式的命名空间注册功能，
让用户可以像调用成员函数一样调用第三方库（如 TA-Lib、bottleneck）。

参考了 polars GitHub issue 中的方案：
https://github.com/pola-rs/polars/issues/9261

使用方法：
expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

参数说明：
skip_nan: bool
    是否跳过空值。可以处理停牌无数据的问题，但会降低运行速度
output_idx: int
    多列输出时，选择只输出其中一列（索引从0开始）
schema: list or tuple
    返回为多列时，会组装成struct，可以提前设置子列的名字
schema_format: str
    为子列名指定格式，例如 'XX_{}_YY' 会将 'upperband' 变成 'XX_upperband_YY'
nan_to_null: bool
    返回值是否将 nan（numpy的空值）转换成 null（polars的空值）

其它参数按**位置参数**和**命名参数**输入皆可，会直接透传给第三方库函数
"""
import numpy as np
from polars import Expr, Float64, DataFrame, api
from polars import Series, Struct


def func_wrap_mn(func, cols,
                 *args,
                 skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                 **kwargs):
    """
    多输入多输出包装函数，兼容单输入单输出场景
    (mn = multiple input multiple output)

    这个函数的核心作用是：
    1. 将 polars 的表达式列转换为 numpy 数组，供第三方库使用
    2. 处理多列输入（struct）和单列输入
    3. 处理多列输出（转成 struct）和单列输出
    4. 可选地处理 nan 值（跳过或转换）

    Parameters
    ----------
    func : callable
        第三方库的函数，如 talib.BBANDS
    cols : Expr
        polars 表达式列，可能是单列或 struct（多列）
    *args : tuple
        传递给第三方库的位置参数
    skip_nan : bool
        是否跳过 nan 值。True 时会将 nan 移到数组开头处理，处理完再移回原位
    output_idx : int or None
        当函数返回多列时，指定只返回第几列（从0开始）
    schema : list or tuple or None
        为多列输出指定列名，如 ['upperband', 'middleband', 'lowerband']
    schema_format : str
        列名格式化字符串，如 'XX_{}_YY' 会将 'upperband' 变成 'XX_upperband_YY'
    nan_to_null : bool
        是否将 numpy 的 nan 转换成 polars 的 null
    **kwargs : dict
        传递给第三方库的关键字参数

    Returns
    -------
    Series or Expr
        返回 polars Series 或 struct 表达式

    """
    # 处理输入列：如果是 struct 类型（多列），则拆分成多个列
    if cols.dtype.base_type() == Struct:
        # 例如：struct(['A', 'B']).ta.AROON
        # 将 struct 中的每个字段提取出来，形成列列表
        _cols = [cols.struct[field] for field in cols.dtype.to_schema()]
    else:
        # 例如：col('A').ta.BBANDS
        # 单列直接包装成列表
        _cols = [cols]

    # 处理 skip_nan 逻辑
    if skip_nan:
        # 将 polars 列转换为 numpy 数组
        _cols = [c.cast(Float64).to_numpy() for c in _cols]
        # 将多个列垂直堆叠成二维数组（多列输入的情况）
        _cols = np.vstack(_cols)

        # 关键技巧：将 nan 值移到数组开头
        # 1. 找出哪些列没有 nan（True 表示该位置没有 nan）
        # 2. argsort 返回排序索引，stable 保持相对顺序
        idx1 = (~np.isnan(_cols).any(axis=0)).argsort(kind='stable')
        # 3. 再次 argsort 得到还原索引（用于将处理后的数据还原到原位置）
        idx2 = idx1.argsort(kind='stable')

        # 按索引重新排列数组，nan 值现在在开头
        _cols = [_cols[i, idx1] for i in range(len(_cols))]
        # 调用第三方库函数处理数据（nan 在开头，第三方库可能忽略）
        result = func(*_cols, *args, **kwargs)

        # 将结果还原到原位置
        if isinstance(result, tuple):
            # 多列输出，每列都要还原
            result = tuple([_[idx2] for _ in result])
        else:
            # 单列输出
            result = result[idx2]
    else:
        # 不处理 nan，直接调用第三方库函数
        result = func(*_cols, *args, **kwargs)

    # 处理输出结果
    if isinstance(result, tuple):
        # 多列输出（如 BBANDS 返回上轨、中轨、下轨）
        if output_idx is None:
            # 没有指定只输出某一列，则将多列组装成 struct
            # 如果没有提供列名，使用默认名称 column_0, column_1, ...
            if schema is None:
                schema = [f'column_{i}' for i in range(len(result))]

            # 应用格式化字符串到列名
            schema = [schema_format.format(name) for name in schema]
            # 将结果转换为 DataFrame，然后转成 struct
            # 注意：nan_to_null 对 struct 中的 nan 无效，因为 struct 内部还是 numpy 数组
            return DataFrame(result, schema=schema, nan_to_null=nan_to_null).to_struct('')
        # 指定了只输出某一列
        if 0 <= output_idx < len(result):
            return Series(result[output_idx], nan_to_null=nan_to_null)
    elif not isinstance(result, Series):
        # 结果不是 Series，需要转换（如 bottleneck 返回 numpy 数组）
        # 例如：col('A').bn.move_rank
        return Series(result, nan_to_null=nan_to_null)
    else:
        # 结果已经是 Series（如 TA-Lib 的某些函数直接返回 Series）
        # 例如：col('A').ta.COS
        if nan_to_null:
            return result.fill_nan(None)  # 将 nan 转换成 null
        else:
            return result


def func_wrap_11(func, cols,
                 *args,
                 skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                 **kwargs):
    """
    单输入单输出包装函数，性能优化版本
    (11 = single input single output)

    相比 func_wrap_mn 的优势：
    1. 只处理单列输入，不需要处理 struct 拆分
    2. 只处理单列输出，不需要处理多列转 struct
    3. 代码更简单，执行速度更快

    适用于：bottleneck 等只接受单列输入、返回单列输出的库

    Parameters
    ----------
    func : callable
        第三方库的函数
    cols : Expr
        polars 表达式列（单列）
    *args, **kwargs
        传递给第三方库的参数
    skip_nan, nan_to_null
        同 func_wrap_mn

    Returns
    -------
    Series
        返回 polars Series
    """
    _cols = cols

    # 处理 skip_nan 逻辑（简化版，只处理单列）
    if skip_nan:
        # 将 polars 列转换为 numpy 数组
        _cols = _cols.cast(Float64).to_numpy()

        # 将 nan 值移到数组开头
        idx1 = (~np.isnan(_cols)).argsort(kind='stable')
        # 还原索引
        idx2 = idx1.argsort(kind='stable')

        _cols = _cols[idx1]
        result = func(_cols, *args, **kwargs)

        # 将结果还原到原位置
        result = result[idx2]
    else:
        # 不处理 nan，直接调用
        result = func(_cols, *args, **kwargs)

    # 处理输出结果
    if isinstance(result, Series):
        # 结果已经是 Series
        if nan_to_null:
            return result.fill_nan(None)  # 将 nan 转换成 null
        else:
            return result
    else:
        # 结果是 numpy 数组，需要转换
        return Series(result, nan_to_null=nan_to_null)


class FuncHelper:
    """
    函数辅助类，实现动态属性访问拦截

    核心机制：
    1. 当用户访问 expr.ta.BBANDS 时，Python 会调用 __getattribute__('BBANDS')
    2. __getattribute__ 从第三方库（如 talib）中获取 BBANDS 函数
    3. 返回一个 lambda 函数，这个 lambda 会调用 polars 的 map_batches
    4. map_batches 会逐批处理数据，调用包装函数（func_wrap_mn 或 func_wrap_11）

    这样用户就可以像调用成员函数一样调用第三方库：
    expr.ta.BBANDS(...)  →  实际调用 talib.BBANDS(...)
    """
    def __init__(self, expr: Expr, lib=None, wrap=None) -> None:
        """
        初始化函数辅助类

        Parameters
        ----------
        expr : Expr
            polars 表达式，如 pl.col('A')
        lib : module
            第三方库模块，如 talib 或 bottleneck
        wrap : callable
            包装函数，用于处理第三方库的输入输出
            - func_wrap_mn: 多输入多输出
            - func_wrap_11: 单输入单输出
        """
        # 使用 object.__setattr__ 避免触发自定义的 __setattr__（如果有的话）
        # 直接调用父类 object 的方法来设置属性
        object.__setattr__(self, '_expr', expr)
        object.__setattr__(self, '_lib', lib)
        object.__setattr__(self, '_wrap', wrap)

    def __getattribute__(self, name: str):
        """
        属性访问拦截方法

        当用户访问 expr.ta.BBANDS 时，Python 会自动调用这个方法
        name 参数就是 'BBANDS'

        Returns
        -------
        function
            返回一个 lambda 函数，该函数会调用 map_batches 处理数据
        """
        # 使用 object.__getattribute__ 获取实例属性
        # 不能直接用 self._expr，因为会递归调用 __getattribute__
        _expr: Expr = object.__getattribute__(self, '_expr')
        _lib = object.__getattribute__(self, '_lib')
        _wrap = object.__getattribute__(self, '_wrap')

        # 从第三方库中获取对应的函数
        # 例如：getattr(talib, 'BBANDS') 返回 talib.BBANDS 函数
        _func = getattr(_lib, name)

        # 返回一个 lambda 函数
        # 这个 lambda 接收用户传入的参数，然后调用 polars 的 map_batches
        return (
            lambda *args, skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False, **kwargs:
            _expr.map_batches(
                # map_batches 会逐批处理数据，每批数据调用这个内部 lambda
                lambda cols: _wrap(_func, cols,
                                   *args,
                                   skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
                                   **kwargs)
            )
        )


@api.register_expr_namespace('ta')
class TaLibHelper(FuncHelper):
    """
    TA-Lib 辅助类，注册到 polars 的 'ta' 命名空间

    使用装饰器 @api.register_expr_namespace('ta') 将此类注册到 polars
    这样用户就可以通过 expr.ta.func() 的方式调用 TA-Lib 函数

    例如：
    pl.col('A').ta.BBANDS(timeperiod=20)
    """
    def __init__(self, expr: Expr) -> None:
        """
        初始化 TA-Lib 辅助类

        Parameters
        ----------
        expr : Expr
            polars 表达式
        """
        import talib as ta
        # 调用父类初始化，传入：
        # - expr: polars 表达式
        # - ta: TA-Lib 库模块
        # - func_wrap_mn: 多输入多输出包装函数（TA-Lib 很多函数支持多列输入输出）
        super().__init__(expr, ta, func_wrap_mn)


@api.register_expr_namespace('bn')
class BottleneckHelper(FuncHelper):
    """
    Bottleneck 辅助类，注册到 polars 的 'bn' 命名空间

    Bottleneck 是一个性能优化的 numpy 函数库
    使用装饰器 @api.register_expr_namespace('bn') 将此类注册到 polars
    这样用户就可以通过 expr.bn.func() 的方式调用 bottleneck 函数

    例如：
    pl.col('A').bn.move_rank(window=20)
    """
    def __init__(self, expr: Expr) -> None:
        """
        初始化 Bottleneck 辅助类

        Parameters
        ----------
        expr : Expr
            polars 表达式
        """
        import bottleneck as bn
        # 调用父类初始化，传入：
        # - expr: polars 表达式
        # - bn: bottleneck 库模块
        # - func_wrap_11: 单输入单输出包装函数（bottleneck 函数通常是单列输入输出）
        super().__init__(expr, bn, func_wrap_11)
