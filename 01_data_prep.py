from datasets import load_dataset
import pandas as pd

# 加载数据集
dataset = load_dataset("dmayhem93/agieval-gaokao-history")

# 假设数据集中有一个 'question' 和 'answer' 字段
# 这里我们将数据集分为训练集和测试集
# 你可以根据需要调整分割比例
train_data = dataset['test'].select(range(100))
test_data = dataset['test'].select(range(100, 200))

# 定义一个映射函数，将数字映射到字母
def map_answer_to_letter(answer):
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    return mapping.get(answer[0], answer)

# 将数据转换为 DataFrame，并映射答案
df_train = pd.DataFrame({
    "question": train_data['query'],
    "answer": [map_answer_to_letter(ans) for ans in train_data['gold']]
})
df_test = pd.DataFrame({
    "question": test_data['query'],
    "answer": [map_answer_to_letter(ans) for ans in test_data['gold']]
})

# 保存为 CSV 文件
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)