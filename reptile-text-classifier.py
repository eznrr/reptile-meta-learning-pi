import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split

# Tokenizador e modelo pré-treinado
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Definindo as categorias fixas das interações terapêuticas
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4, 
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}

# Definição das classes e exemplos
dataset = [
    # Aprovação (APR)
    ("Com certeza! Você tem toda a razão.", "APR"),
    ("Você está completamente certo.", "APR"),
    ("Concordo plenamente com o que você disse.", "APR"),
    ("Sem dúvida, isso faz total sentido.", "APR"),
    ("Exatamente! Você acertou.", "APR"),
    ("Eu entendo completamente sua perspectiva.", "APR"),
    ("Sim, concordo totalmente com isso.", "APR"),
    ("Não tenho dúvida de que é isso mesmo.", "APR"),
    ("Concordo com você, é isso.", "APR"),
    ("Você está correto, sem sombra de dúvida.", "APR"),

    # Empatia (EMP)
    ("Imagino o quanto isso te deixa ansioso.", "EMP"),
    ("Deve ser realmente difícil para você.", "EMP"),
    ("Posso entender como isso pode ser frustrante.", "EMP"),
    ("Isso parece estar te deixando muito sobrecarregado.", "EMP"),
    ("Sei como deve ser complicado passar por isso.", "EMP"),
    ("Compreendo o que você está sentindo.", "EMP"),
    ("Isso deve ser realmente desafiador para você.", "EMP"),
    ("Imagino como você se sente agora.", "EMP"),
    ("Deve estar sendo bem difícil para você lidar com isso.", "EMP"),
    ("Sei que não é fácil, mas você está indo bem.", "EMP"),

    # Facilitação (FAC)
    ("Hum hum.", "FAC"),
    ("Entendi, continue falando.", "FAC"),
    ("Pode seguir com o que você estava dizendo.", "FAC"),
    ("Pode continuar, estou ouvindo.", "FAC"),
    ("Isso é interessante, me conte mais.", "FAC"),
    ("Entendi, pode prosseguir.", "FAC"),
    ("Você está no caminho certo, continue.", "FAC"),
    ("Isso é bom, fale mais sobre isso.", "FAC"),
    ("Fale mais sobre o que você está pensando.", "FAC"),
    ("Você está bem, continue.", "FAC"),

    # Informação (INF)
    ("Biologia requer vários cursos adicionais de laboratório.", "INF"),
    ("Para seguir na área de biologia, é preciso muita prática.", "INF"),
    ("Sabia que a biologia envolve muitos experimentos e laboratórios?", "INF"),
    ("Na biologia, é essencial fazer muitos cursos de laboratório.", "INF"),
    ("A biologia exige que você tenha uma base prática forte.", "INF"),
    ("Para aprender biologia, é necessário se aprofundar nos experimentos.", "INF"),
    ("A biologia vai muito além da teoria, é importante a prática em laboratório.", "INF"),
    ("Cursos de laboratório são fundamentais para a formação em biologia.", "INF"),
    ("A biologia exige cursos práticos para compreensão completa.", "INF"),
    ("Não podemos esquecer que a biologia é uma área muito prática.", "INF"),

    # Interpretação (INT)
    ("Talvez o seu problema não seja de motivação, mas que até agora as coisas ainda não deram certo.", "INT"),
    ("Pode ser que o que está acontecendo seja uma série de fatores, e não apenas falta de motivação.", "INT"),
    ("Eu acho que o problema pode estar mais em como as coisas têm acontecido, e não na sua motivação.", "INT"),
    ("Talvez o desafio seja mais sobre o contexto e as circunstâncias do que motivação em si.", "INT"),
    ("Eu vejo que pode ser mais uma questão de timing, não de motivação.", "INT"),
    ("O que você está enfrentando pode ser mais sobre a situação, e não sobre sua vontade de melhorar.", "INT"),
    ("Às vezes as coisas não acontecem como esperamos, não é apenas falta de motivação.", "INT"),
    ("O que você descreveu pode ter mais a ver com as circunstâncias do que com a motivação.", "INT"),
    ("Talvez o seu problema não seja um problema interno, mas sim algo no ambiente ao seu redor.", "INT"),
    ("Pode ser que o problema seja mais externo do que você imagina, não está relacionado à sua motivação.", "INT"),

    # Outras vocal terapeuta (OVT)
    ("Aceita uma balinha?", "OVT"),
    ("Gostaria de um copo d'água?", "OVT"),
    ("Quer descansar um pouco?", "OVT"),
    ("Posso te oferecer algo para beber?", "OVT"),
    ("Que tal uma pausa rápida para relaxar?", "OVT"),
    ("Você gostaria de algo para comer?", "OVT"),
    ("Que tal uma pausa antes de continuar?", "OVT"),
    ("Posso te ajudar com alguma coisa agora?", "OVT"),
    ("Você quer algo para se sentir mais confortável?", "OVT"),
    ("Sente-se à vontade para pedir o que precisar.", "OVT"),

    # Recomendação (REC)
    ("Tente conversar com seu pai durante a semana e lhe falar sobre o que você sente nessas situações.", "REC"),
    ("Acho que você poderia tentar abrir o coração com seus pais sobre isso.", "REC"),
    ("Uma boa ideia seria conversar mais com seu pai sobre o que você sente.", "REC"),
    ("Que tal falar com seu pai durante a semana sobre isso?", "REC"),
    ("Seria bom se você compartilhasse seus sentimentos com seu pai.", "REC"),
    ("Tente se abrir mais com seu pai e expressar como se sente.", "REC"),
    ("Que tal uma conversa com seu pai sobre o que você está vivendo?", "REC"),
    ("Converse com seu pai e explique o que está acontecendo com você.", "REC"),
    ("Tente ser mais transparente com seu pai sobre o que você está sentindo.", "REC"),
    ("Acredito que uma conversa com seu pai pode ajudar a entender melhor suas emoções.", "REC"),

    # Solicitação de relato (SREL)
    ("Me conta... Por que é que você está procurando terapia?", "SREL"),
    ("O que te motivou a buscar a terapia agora?", "SREL"),
    ("Conta mais sobre o que te trouxe até aqui.", "SREL"),
    ("Como você chegou à conclusão de que precisava de terapia?", "SREL"),
    ("O que te fez decidir procurar ajuda terapêutica?", "SREL"),
    ("Quais são as razões que te fizeram procurar a terapia?", "SREL"),
    ("O que aconteceu para que você decidisse começar a terapia?", "SREL"),
    ("Fale um pouco sobre o que te trouxe até esse ponto.", "SREL"),
    ("O que aconteceu para que você decidisse começar a terapia?", "SREL"),
    ("Pode me contar mais sobre o que te fez procurar ajuda?", "SREL"),

    # Solicitação de reflexão (SREF)
    ("Você tem alguma hipótese de por que isso aconteceu?", "SREF"),
    ("O que você acha que pode ter causado essa situação?", "SREF"),
    ("Já pensou nas possíveis razões para isso estar acontecendo?", "SREF"),
    ("O que você acha que levou a esse resultado?", "SREF"),
    ("Você consegue refletir sobre o que pode ter ocasionado isso?", "SREF"),
    ("Qual é a sua opinião sobre o que pode ter causado essa situação?", "SREF"),
    ("Por que você acha que isso aconteceu? Já refletiu sobre isso?", "SREF"),
    ("O que pensa ser o motivo disso?", "SREF"),
    ("Como você explicaria esse acontecimento?", "SREF"),
    ("O que você considera ser a causa disso?", "SREF"),

    # Reprovação (REP)
    ("Eu não acho que seja assim.", "REP"),
    ("Não concordo com essa visão.", "REP"),
    ("Essa não é a melhor maneira de ver as coisas.", "REP"),
    ("Eu vejo de uma forma diferente.", "REP"),
    ("Não acredito que seja essa a solução.", "REP"),
    ("Eu não vejo dessa forma, acho que há outra perspectiva.", "REP"),
    ("Não posso concordar com isso.", "REP"),
    ("Não acho que esse seja o caminho.", "REP"),
    ("Eu discordo dessa ideia.", "REP"),
    ("Não creio que seja por aí.", "REP")
]

# Transformar frases em embeddings
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # Extraindo embedding do token [CLS]

# Criar um dataset com embeddings e mapeamento para índices
data = [(get_embedding(text), categorias_interacoes[label]) for text, label in dataset]

def sample_task(data, num_classes=5, num_examples=5):
    """Cria uma tarefa few-shot selecionando classes e exemplos aleatórios."""
    selected_labels = random.sample(list(categorias_interacoes.values()), num_classes)
    task_data = [
        (embed, label) for embed, label in data if label in selected_labels
    ]
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
def train_reptile(model, data, num_tasks=1000, inner_steps=5, meta_lr=0.1):
    """Treinamento do modelo usando Reptile Meta-Learning."""
    for task in range(num_tasks):
        model_copy = Classifier(embedding_dim, num_classes)
        model_copy.load_state_dict(model.state_dict())
        optimizer_task = optim.Adam(model_copy.parameters(), lr=1e-2)

        task_data = sample_task(data)
        X_train, y_train = zip(*task_data)
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train)

        # Treinamento interno para tarefa few-shot
        for step in range(inner_steps):
            optimizer_task.zero_grad()
            loss = nn.CrossEntropyLoss()(model_copy(X_train), y_train)
            loss.backward()
            optimizer_task.step()

        # Atualização meta com médias de gradientes (Reptile Update)
        with torch.no_grad():
            for param, param_copy in zip(model.parameters(), model_copy.parameters()):
                param.data += meta_lr * (param_copy.data - param.data)

        if task % 100 == 0:
            print(f"Tarefa {task}/{num_tasks} concluída.")

# Função para salvar o modelo treinado
def salvar_modelo(model, caminho_arquivo):
    torch.save(model.state_dict(), caminho_arquivo)
    print(f"Modelo salvo em {caminho_arquivo}")

# Executar treinamento e salvar o modelo treinado
train_reptile(model, data)
salvar_modelo(model, "modelo_treinado_reptile.pth")
print("Treinamento e salvamento concluídos!")