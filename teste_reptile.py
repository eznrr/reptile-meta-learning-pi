import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# === Parâmetros ===
embedding_dim = 768
num_classes_total = 10
modelo_path = "modelo_treinado_reptile.pth"

# === Categorias ===
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4, 
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}
idx_to_categoria = {v: k for k, v in categorias_interacoes.items()}

# === Modelo de Classificação ===
class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# === Carregar Modelo e Tokenizador ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

model = Classifier(embedding_dim, num_classes_total)
model.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
model.eval()

# === Função para montar o texto ===
def montar_texto(fala_cliente, fala_terapeuta):
    if fala_cliente.strip():
        return f"Cliente: {fala_cliente} Terapeuta: {fala_terapeuta}"
    else:
        return f"Terapeuta: {fala_terapeuta}"

# === Função para gerar embedding ===
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

# === Loop de Teste ===
print("\nDigite a fala do terapeuta para classificar. Você pode também inserir a fala do cliente para dar contexto.")
print("Deixe a fala do cliente vazia se quiser testar sem contexto.")
print("Digite 'sair' como fala do terapeuta para encerrar.\n")

while True:
    fala_cliente = input("Fala do cliente: ").strip()
    fala_terapeuta = input("Fala do terapeuta: ").strip()

    if fala_terapeuta.lower() == "sair":
        break

    texto = montar_texto(fala_cliente, fala_terapeuta)
    embedding = get_embedding(texto).unsqueeze(0)

    with torch.no_grad():
        output = model(embedding)
        pred = torch.argmax(output, dim=1).item()

    print(f"\nTexto usado para predição:\n{texto}")
    print(f"Categoria prevista: {idx_to_categoria[pred]}\n{'-'*50}")
