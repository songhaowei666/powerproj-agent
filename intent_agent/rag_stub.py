"""RAG 查询预留接口，后续接入真实向量检索。"""

from typing import List, Dict


async def retrieve_similar_examples(query: str, k: int = 3) -> List[Dict]:
    """从 RAG 检索与用户 query 语义最相似的少样本示例。

    Args:
        query: 用户输入的查询语句
        k: 返回样本数量

    Returns:
        示例列表，每个元素包含 query 和 tasks 字段

    TODO: 接入真实向量数据库检索
    """
    return []
