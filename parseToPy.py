input_path = r'I:\GitProject\controllableRag8.13\sophisticated_rag_agent_harry_potter.ipynb'
output_path = r'I:\GitProject\controllableRag8.13\output.txt'

import nbformat

def extract_notebook_to_text(filename, output_filename):
    # 读取notebook文件
    with open(filename, 'r', encoding='utf-8') as file:
        nb = nbformat.read(file, as_version=4)

    # 准备收集所有文本
    all_text = []

    # 遍历每一个单元格
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 收集代码单元格的内容
            all_text.append("# Code Cell\n" + cell.source + "\n")
        elif cell.cell_type == 'markdown':
            # 收集Markdown单元格的内容
            all_text.append("# Markdown Cell\n" + cell.source + "\n")

    # 将所有文本保存到一个文件中
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(all_text))

# 替换以下路径为您的具体文件路径
input_path = r'I:\GitProject\controllableRag8.13\sophisticated_rag_agent_harry_potter.ipynb'


output_path = r'I:\GitProject\controllableRag8.13\output.txt'
extract_notebook_to_text(input_path, output_path)

