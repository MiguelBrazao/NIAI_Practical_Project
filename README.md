# NIAI Practical Project

Nature-Inspired Artificial Intelligence Practical Project. The primary objective of this project is to design and implement an Evolutionary Algorithm (EA) capable of evolving autonomous controllers for Super Mario.

---

## 1. Criar o Ambiente Anaconda

> Apenas necessário na primeira vez.

Abrir o **Anaconda Prompt** e executar os seguintes comandos:

```bash
# Criar o ambiente com Python 3.10
conda create -n NIAI python=3.10

# Ativar o ambiente
conda activate NIAI

# Navegar para a pasta code/ do projeto
cd /d "<caminho_para_a_pasta_code>"

# Instalar as dependências
python install_requirements.py
```

> **Nota:** O `install_requirements.py` é um script Python que instala automaticamente todas as dependências via `pip` (`torch`, `numpy`, `matplotlib`, `deap`).

---

## 2. Correr o Projeto

### Passo 1 — Iniciar o ambiente Docker

Abrir um terminal na pasta `code/` e executar:

```bash
docker compose up
```

Depois abrir o browser e ir a http://localhost para ver o ambiente do jogo.

### Passo 2 — Correr o script Python

Abrir o **Anaconda Prompt**, ativar o ambiente e navegar até à pasta `code/`:

```bash
conda activate NIAI

cd /d "<caminho_para_a_pasta_code>"
```

Depois correr um dos scripts (o `42` é a seed, pode ser qualquer número):

```bash
# Usar Genetic Programming
python mario_random_search_gp.py 42

# Usar MLP (Multi-Layer Perceptron)
python mario_random_search_mlp.py 42
```