# Detecção de Anomalias Cardiológicas em ECG (PTB-XL) — 2025.1

Este repositório contém o desenvolvimento de um pipeline de **classificação de ECG** usando a base **PTB-XL** (Kaggle) e modelos de **Aprendizado de Máquina/Deep Learning**. O notebook principal implementa:
- preparo do dataset,
- pré-processamento dos sinais,
- extração de características (FFT, DWT, entropia de Shannon e estatísticas no tempo),
- treinamento de **CNN 1D** e **MLP** (Testes 4 e 7),
- avaliação (matrizes de confusão, curvas ROC, análise de overfitting),
- comparação com baseline (ex.: variação “FFT-only”).

## 🎯 Objetivo
Classificar registros de ECG em três classes:
- **MI** — Infarto Agudo do Miocárdio  
- **Abn-HB** — Batimento Cardíaco Anormal (Arritmias)  
- **NORM** — ECG Normal

> Essas classes aparecem no início do notebook, que também justifica a escolha clínica e a importância de incluir a classe “normal”.

## 📂 Estrutura do repositório
```
Projeto_Sinais_2025_1.ipynb     # Notebook principal (renomear se necessário)
README.md                       # Este arquivo
```

Durante a execução no Colab, o notebook cria/usa pastas e arquivos como:
```
/content/ptb-xl-dataset/...                  # dados baixados do Kaggle
/content/splited_dataset/
    X_signals.npy
    y_labels.npy
    X_filtered.npy
    y_filtered.npy
/content/cnn_metrics/                        # métricas/artefatos dos testes (seções de treino)
```

## 🧰 Dependências principais
Executar no **Google Colab** é recomendado. Bibliotecas usadas no notebook:
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- **Sinais/ECG**: `wfdb`, `pywt` (Wavelets)
- **ML/DL**: `tensorflow/keras`, `scikit-learn`
- Utilitários: `pydub`, `zipfile`, `shutil`, `pickle`
- **Kaggle API** (para baixar a PTB-XL diretamente no Colab)

## ▶️ Como executar (Colab)
1. **Abra o notebook no Colab**  
   [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/repositorio/blob/main/Projeto_Sinais_2025_1.ipynb)  

2. **Configure a Kaggle API** (primeiras células da seção “Importing Dataset from Kaggle”):  
   - Faça upload do `kaggle.json` no Colab (Conta Kaggle → API → Create New Token).  
   - Rode as células que copiam o token para `~/.kaggle` e definem permissões.  
   - Execute o download:
     ```
     kaggle datasets download -d khyeh0719/ptb-xl-dataset
     unzip -o ptb-xl-dataset.zip -d /content/ptb-xl-dataset
     ```
3. **Processe os dados** seguindo as células da seção “Processamento da Base PTB-XL” / “Pré-processamento dos sinais”.  
   O notebook gera arrays `.npy` em `/content/splited_dataset/` para acelerar experimentos.

4. **Rode os Testes (1–7)**  
   Cada “Teste” configura uma variação (ex.: arquitetura/hiperparâmetros/feature set).  
   Ao final, o notebook plota **curvas de treino/validação**, **matrizes de confusão**, **curvas ROC** e faz **análise textual** (overfitting, ponto de parada, implicações).

## 🔧 Pipeline (resumo)
- **Pré-processamento**  
  Normalização (`StandardScaler` por canal), estabilidade de amostras, janelamento dos sinais (entrada típica reportada no notebook: **407×12** amostras×derivações, podendo variar conforme o teste).
- **Aumento de dados (Data Augmentation)**  
  Ruído gaussiano, **time-shift** (deslocamento temporal), **escala de amplitude** e **inversão** opcional.
- **Extração de características**  
  - **FFT** (espectro por derivação, compressão log, seleção de bins)  
  - **DWT** (ex.: Daubechies, níveis configuráveis; coeficientes por canal)  
  - **Entropia de Shannon** (em janelas/espectro)  
  - Estatísticas no tempo (ex.: média, variância, **skewness**, **kurtosis**)
- **Modelo principal — CNN 1D**  
  Arquitetura típica descrita nas seções de teste: pilha de `Conv1D` + `BatchNorm` + `LeakyReLU` + `Dropout`, seguida de `GlobalAveragePooling1D` e `Dense` final.  
  *Treino com* **EarlyStopping**, `StratifiedKFold`/hold-out e **class weights** para lidar com desbalanceamento.
- **Baseline/Comparativos**  
  Versão **FFT-only** e variações; comparação qualitativa e via métricas.

## 📈 Avaliação
- **Matrizes de confusão** por teste/classe  
- **Curvas ROC** (one-vs-rest) com **AUC**  
- Curvas **Loss/Accuracy** (treino × validação), análise de **overfitting** e ponto de **early stopping**  
- Comentários sobre **generalização** e **viés** das abordagens

## 🔁 Reprodutibilidade
O notebook fixa seeds (`numpy`, `random`, `tensorflow`) em várias seções. A ordem e a **execução sequencial** das células é importante para replicar os resultados.

## 👥 Equipe
- Vinícius de Sousa Rodrigues (vsr)  
- Álvaro Cavalcante Negromonte (acn3)  
- Júlio César Barbosa da Silva (jcbs3)  
- Luiz Felipe Silva Lustosa (lfsl)  
- Felipe Torres de Macedo (ftm2)  
- Edinaldo Filho (ebcf2)

> **Observação ética**: O projeto usa a base pública PTB-XL e é voltado a fins acadêmicos. Resultados não devem ser usados como ferramenta diagnóstica clínica.
