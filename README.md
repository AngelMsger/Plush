# Plush

《集体智慧编程》中的推荐、聚类和搜索示例。

## 环境

使用 Python 3.12（CI 验证版本）。当前 NumPy/SciPy 依赖要求 Python 3.12 或更新版本。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

依赖包含搜索示例所需的 Beautiful Soup 和 jieba；Requests、urllib3 和 idna
使用相互兼容的版本，Pillow 与科学计算库使用支持该 Python 版本的版本。

## 验证

```sh
python -m unittest discover -s tests -v
```

测试覆盖推荐计算、不同大小和空簇的 K 均值聚类、Pillow 绘图，以及通过本地
HTTP 服务抓取、分词并写入 SQLite。测试不会访问外部网站。
