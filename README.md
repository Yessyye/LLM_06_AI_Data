# 自动数据分析报告生成系统

这是一个自动化数据分析报告生成系统。用户上传一个 CSV 文件，系统会自动生成数据分析报告，报告包括：
- 数据的基本描述性统计
- 数据清洗和处理
- 探索性数据分析（EDA）
- 模型推荐和评估指标

## 使用说明

1. 安装所需依赖：
    ```bash
    pip install -r requirements.txt
    ```

2. 运行 Flask 应用：
    ```bash
    python app.py
    ```

3. 打开浏览器并访问 `http://127.0.0.1:5000/`，上传 CSV 文件进行分析。

4. 上传成功后，报告会自动生成，并提供下载链接（Markdown 和 PDF 格式）。

## 配置

- 请在 `app.py` 中填入你的 OpenAI API 密钥：`openai.api_key = 'YOUR_OPENAI_API_KEY'`
