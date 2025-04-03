import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# Tokenizador e modelo pré-treinado
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Definindo as categorias fixas das interações terapêuticas
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4, 
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}

csv_file = "classificacao_simccit.csv"
df = pd.read_csv(csv_file)
dataset = list(zip(df["Interação"].tolist(), df["Categoria"].tolist()))

# Transformar frases em embeddings
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # Extraindo embedding do token [CLS]

# Criar um dataset com embeddings e mapeamento para índices
data = [(get_embedding(text), categorias_interacoes[label]) for text, label in dataset]

# Separar em treino e teste
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def sample_task(data, num_classes=5, num_examples=10):
    """Cria uma tarefa few-shot selecionando classes e exemplos aleatórios."""
    selected_labels = random.sample(list(categorias_interacoes.values()), num_classes)
    task_data = [(embed, label) for embed, label in data if label in selected_labels]
    return random.sample(task_data, num_examples * num_classes)

# Definição do modelo de classificação simples
class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Configuração do modelo e hiperparâmetros
embedding_dim = 768
num_classes = len(categorias_interacoes)
model = Classifier(embedding_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Função de treinamento com Reptile
def train_reptile(model, data, num_tasks=2000, inner_steps=20, meta_lr=0.1):
    """Treinamento do modelo usando Reptile Meta-Learning."""
    for task in range(num_tasks):
        model_copy = Classifier(embedding_dim, num_classes)
        model_copy.load_state_dict(model.state_dict())
        optimizer_task = optim.Adam(model_copy.parameters(), lr=1e-2)

        task_data = sample_task(data)
        X_train, y_train = zip(*task_data)
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train)

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

# Função para salvar o modelo treinado
def salvar_modelo(model, caminho_arquivo):
    torch.save(model.state_dict(), caminho_arquivo)
    print(f"Modelo salvo em {caminho_arquivo}")

# Função de avaliação
def avaliar_modelo(model, data, num_test_samples=200):
    model.eval()  # Modo de avaliação
    test_data = random.sample(data, min(num_test_samples, len(data)))
    X_test, y_test = zip(*test_data)
    X_test = torch.stack(X_test)
    y_test = torch.tensor(y_test)

    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1)
        corretas = (preds == y_test).sum().item()
        total = y_test.size(0)
        acuracia = corretas / total

    print(f"Acurácia do modelo em {total} exemplos de teste: {acuracia:.2%}")
    return acuracia

# Executar treinamento e salvar o modelo treinado
train_reptile(model, train_data)
salvar_modelo(model, "modelo_treinado_reptile.pth")

# Avaliação do modelo
avaliar_modelo(model, test_data)
