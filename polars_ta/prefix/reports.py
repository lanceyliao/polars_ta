# 只导出报告类指标本身，不星号带入 wq 依赖与 Expr，避免污染默认命名空间。
from polars_ta.reports.cicc import ts_RSRS  # noqa
from polars_ta.reports.cicc import ts_RSRS_R2  # noqa
