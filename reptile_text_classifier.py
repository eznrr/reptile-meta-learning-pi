import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

# Configurações e hiperparâmetros
embedding_dim = 768
num_classes_total = 10
num_tasks = 3000
inner_steps = 20
meta_lr = 0.01
num_classes_por_tarefa = 10
num_exemplos_por_classe = 6

# Tokenizador e modelo BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Mapeamento de categorias
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4, 
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}

# Leitura dos dados com contexto (Fala_Cliente + Fala_Terapeuta)
csv_file = "classificacao_simccit.csv"
df = pd.read_csv(csv_file)
df = df.dropna(subset=["Fala_Terapeuta", "Categoria"])
df["Fala_Cliente"] = df["Fala_Cliente"].fillna("")

# Construção do texto com contexto e flag de contexto
def montar_texto(row):
    if row['Fala_Cliente'].strip():
        return f"Cliente: {row['Fala_Cliente']} Terapeuta: {row['Fala_Terapeuta']}", 1
    else:
        return f"Terapeuta: {row['Fala_Terapeuta']}", 0

df[['Interacao', 'Tem_Contexto']] = df.apply(montar_texto, axis=1, result_type='expand')
dataset = list(zip(df["Interacao"].tolist(), df["Categoria"].tolist(), df['Tem_Contexto'].tolist()))
random.shuffle(dataset)

# Geração dos embeddings
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # Embedding do token [CLS]

# Embeddings e rótulos numéricos
data = [(get_embedding(text), categorias_interacoes[label], tem_contexto) for text, label, tem_contexto in dataset if label in categorias_interacoes]

# Divisão treino/teste com shuffle
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Função few-shot balanceada
def sample_task_balanceado(data, num_classes=10, num_examples=20):
    por_classe = defaultdict(list)
    for embed, label, _ in data:
        por_classe[label].append(embed)

    selected_labels = random.sample(list(por_classe.keys()), num_classes)
    task_data = []
    for label in selected_labels:
        exemplos = por_classe[label]
        if len(exemplos) < num_examples:
            escolhidos = random.choices(exemplos, k=num_examples)
        else:
            escolhidos = random.sample(exemplos, num_examples)
        task_data.extend([(e, label) for e in escolhidos])

    return task_data

# Modelo de classificação simples
class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Treinamento com Reptile
def train_reptile(model, data, num_tasks, inner_steps, meta_lr):
    freq_classes = Counter()

    for task in range(num_tasks):
        model_copy = Classifier(embedding_dim, num_classes_total)
        model_copy.load_state_dict(model.state_dict())
        optimizer_task = optim.Adam(model_copy.parameters(), lr=1e-2)

        task_data = sample_task_balanceado(data, num_classes=num_classes_por_tarefa, num_examples=num_exemplos_por_classe)
        X_train, y_train = zip(*task_data)
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train)

        freq_classes.update(y_train.tolist())

        for step in range(inner_steps):
            optimizer_task.zero_grad()
            loss = nn.CrossEntropyLoss()(model_copy(X_train), y_train)
            loss.backward()
            optimizer_task.step()

        with torch.no_grad():
            for param, param_copy in zip(model.parameters(), model_copy.parameters()):
                param.data += meta_lr * (param_copy.data - param.data)

        if task % 100 == 0:
            print(f"Tarefa {task}/{num_tasks} concluída.")

    print("\nFrequência de classes durante o treinamento:")
    for idx, count in sorted(freq_classes.items()):
        nome = [k for k, v in categorias_interacoes.items() if v == idx][0]
        print(f"{nome} ({idx}): {count} exemplos")

# Função para salvar o modelo
def salvar_modelo(model, caminho_arquivo):
    torch.save(model.state_dict(), caminho_arquivo)
    print(f"\nModelo salvo em {caminho_arquivo}")

# Avaliação completa com relatório e matriz de confusão + separação por contexto
def avaliar_modelo_completo(model, data, categorias_map, num_test_samples=200):
    model.eval()
    test_data = random.sample(data, min(num_test_samples, len(data)))
    X_test, y_true, contexto_flags = zip(*test_data)
    X_test = torch.stack(X_test)
    y_true = torch.tensor(y_true)

    with torch.no_grad():
        outputs = model(X_test)
        y_pred = torch.argmax(outputs, dim=1)

    idx_to_label = {v: k for k, v in categorias_map.items()}
    nomes_categorias = [idx_to_label[i] for i in sorted(idx_to_label)]

    print("\nRelatório de Classificação Geral:")
    print(classification_report(y_true, y_pred, target_names=nomes_categorias, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=nomes_categorias, yticklabels=nomes_categorias, cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão Geral")
    plt.tight_layout()
    plt.show()

    # Separar com e sem contexto
    com_ctx = [(x, y, p) for x, y, p, c in zip(X_test, y_true, y_pred, contexto_flags) if c == 1]
    sem_ctx = [(x, y, p) for x, y, p, c in zip(X_test, y_true, y_pred, contexto_flags) if c == 0]

    for nome, grupo in [("Com Contexto", com_ctx), ("Sem Contexto", sem_ctx)]:
        if len(grupo) == 0:
            print(f"\nNenhum exemplo '{nome.lower()}'.")
            continue
        _, y_ref, y_pd = zip(*grupo)
        print(f"\nRelatório de Classificação - {nome}:")
        print(classification_report(y_ref, y_pd, target_names=nomes_categorias, digits=3))

# ===== Execução =====
model = Classifier(embedding_dim, num_classes_total)
train_reptile(model, train_data, num_tasks=num_tasks, inner_steps=inner_steps, meta_lr=meta_lr)
salvar_modelo(model, "modelo_treinado_reptile.pth")
avaliar_modelo_completo(model, test_data, categorias_interacoes)
