#!/usr/bin/env python3
"""
使用 ModelScope 下载离线模型脚本。
模型: QuantFactory/gorilla-openfunctions-v2-GGUF
"""

import os
from modelscope import snapshot_download

# 配置
MODEL_ID = "QuantFactory/gorilla-openfunctions-v2-GGUF"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    print(f"开始下载模型: {MODEL_ID}")
    print(f"下载目录: {CACHE_DIR}")

    try:
        model_dir = snapshot_download(
            model_id=MODEL_ID,
            cache_dir=CACHE_DIR,
        )
        print(f"模型下载完成，保存路径: {model_dir}")
    except Exception as e:
        print(f"下载失败: {e}")
        raise


if __name__ == "__main__":
    main()
