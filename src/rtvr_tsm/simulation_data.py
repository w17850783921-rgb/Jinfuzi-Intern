#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pymysql
import logging
from datetime import date, datetime
import traceback
from typing import Union, List, Dict
import sys

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # src目录
project_root = os.path.dirname(current_dir)  # 项目根目录

# 将项目根目录添加到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 添加到开头，优先搜索

try:
    # 现在可以正常导入了
    from utils.data_fetcher import DataFetcher
    from utils.data_fetcher_jy_simulation import DataFetcher_jy

    print("成功导入 DataFetcher 和 DataFetcher_jy!")
except ImportError as e:
    print(f"导入失败，使用备用类。错误信息: {e}")

    # 备用类
    class DataFetcher:
        def __init__(self, data_dir: str):
            pass

        def trade_cal(self, exchange: str, start_date: str, end_date: str, is_open: str):
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            df = pd.DataFrame(index=dates)
            return df



if __name__ == "__main__":
    data_dir = './data/simulation_data'
    os.makedirs(data_dir, exist_ok=True)

    fetcher = DataFetcher_jy(data_dir=data_dir)
    try:
        results = fetcher.fetch_a_index_weight_and_return(
            index_codes=[
                '000905.SH',
                '000985',
                '000922.SH',
                '512510',
                '515180'
            ],
            start_date='20201228',
            end_date='20261231'
        )

        # 打印结果概览
        for code, df in results.items():
            print(f"\n📊 {code}: {df.shape} | 列: {list(df.columns)[:5]}...")

        # 合并指定文件
        fetcher.merge_specified_files([
            "idx_000905_SH_return.csv",
            "idx_000922_SH_return.csv",
            "idx_000985_return.csv",
            "fund_512510_return.csv",
            "fund_515180_return.csv"
        ])

    finally:
        fetcher.close()