import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

# Definir categorias de interações terapêuticas
categorias_interacoes = {
    0: "APR", 1: "EMP", 2: "FAC", 3: "INF", 4: "INT", 
    5: "OVT", 6: "REC", 7: "SREL", 8: "SREF", 9: "REP"
}

# Carregar tokenizador e modelo treinado
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Carregar modelo treinado
embedding_dim = 768
num_classes = len(categorias_interacoes)
model = Classifier(embedding_dim, num_classes)
model.load_state_dict(torch.load("modelo_treinado_reptile.pth"))
model.eval()  # Colocar o modelo em modo de avaliação

# Função para gerar o embedding de uma frase
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = DistilBertModel.from_pretrained("distilbert-base-uncased")(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # Extraindo o token [CLS]

# Função para prever a categoria da interação
def prever_categoria(frase):
    embedding = get_embedding(frase).unsqueeze(0)  # Adiciona batch dimension
    logits = model(embedding)
    predicao = torch.argmax(logits, dim=1).item()  # Obtém a classe prevista
    return categorias_interacoes[predicao]

# Exemplo de teste
frase_teste = "Talvez voce tenha que pensar no que fez para que isso acontecesse, nao?"
categoria_prevista = prever_categoria(frase_teste)

print(f"Frase: {frase_teste}")
print(f"Classificada como: {categoria_prevista}")
