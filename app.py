import os
import openai
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 配置 OpenAI API 密钥
openai.api_key = 'YOUR_OPENAI_API_KEY'

app = Flask(__name__)

# 确保文件上传目录存在
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 首页：上传文件
@app.route('/')
def index():
    return render_template('index.html')


# 处理文件上传并生成报告
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 加载 CSV 数据
        df = pd.read_csv(file_path)

        # 调用 OpenAI 生成分析报告
        report_text, charts = generate_report(df)

        # 保存报告为 Markdown
        report_filename_md = os.path.join(REPORT_FOLDER, 'Data-Analysis-Report.md')
        with open(report_filename_md, 'w') as f:
            f.write(report_text)

        # 保存报告为 PDF
        report_filename_pdf = os.path.join(REPORT_FOLDER, 'Data-Analysis-Report.pdf')
        save_pdf_report(report_text, charts, report_filename_pdf)

        return render_template('index.html', message="报告生成成功！", report_md='Data-Analysis-Report.md',
                               report_pdf='Data-Analysis-Report.pdf')
    return "Invalid file format. Please upload a CSV file."


# 使用 OpenAI 生成数据分析报告
def generate_report(df):
    # 生成报告内容
    prompt = f"""
    这是一个数据集，以下是该数据集的描述：
    {df.describe()}

    请进行以下分析：
    1. 检查缺失值并给出处理建议。
    2. 进行基本的探索性数据分析（EDA）。
    3. 给出适合的模型推荐和评估指标。

    生成的报告需要包括数据的描述性统计、可视化图表和模型建议。
    """

    # 使用 OpenAI 生成初步的分析报告
    response = openai.Completion.create(
        model="gpt-4",  # 使用 GPT-4
        prompt=prompt,
        temperature=0.7,
        max_tokens=1500
    )

    report_text = response.choices[0].text.strip()

    # 可视化分析
    charts = generate_charts(df)

    # 模型训练与评估
    model_summary = train_and_evaluate_model(df)

    # 将模型评估结果添加到报告
    report_text += "\n\n## 模型评估\n" + model_summary

    return report_text, charts


# 生成数据可视化图表
def generate_charts(df):
    charts = []

    # 直方图
    plt.figure(figsize=(8, 6))
    df.hist(bins=30, figsize=(10, 8))
    plt.tight_layout()
    hist_path = os.path.join(REPORT_FOLDER, 'histogram.png')
    plt.savefig(hist_path)
    charts.append(hist_path)
    plt.close()

    # 相关性热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.tight_layout()
    corr_path = os.path.join(REPORT_FOLDER, 'correlation_heatmap.png')
    plt.savefig(corr_path)
    charts.append(corr_path)
    plt.close()

    return charts


# 模型训练与评估
def train_and_evaluate_model(df):
    # 假设目标列是 'target_column'，如果没有该列，可以改为其他列名
    if 'target_column' not in df.columns:
        return "没有找到目标列（'target_column'），无法进行模型评估。"

    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # 对类别变量进行编码（如果有的话）
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练分类模型（RandomForestClassifier）
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return f"分类模型的准确率：{accuracy:.4f}"


# 将报告保存为 PDF
def save_pdf_report(report_text, charts, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="数据分析报告", ln=True, align="C")
    pdf.multi_cell(0, 10, report_text)

    # 添加图表
    for chart in charts:
        pdf.add_page()
        pdf.image(chart, x=10, y=10, w=180)

    pdf.output(output_path)


# 提供生成的报告供下载
@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
