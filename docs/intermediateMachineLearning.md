# 📊 Machine Learning Intermediário — Um Guia Prático e Comentado
<br>

# 📘 Capítulo 1 — Introduction

<details>
<br>

> ### *Machine Learning Intermediário — Um Guia Prático e Comentado*

---

## 🟦 1.1. Objetivo do Curso

<details>
<br>

O curso **Kaggle Intermediate Machine Learning** tem como objetivo aprofundar conceitos essenciais de pré‑processamento e modelagem, indo além do treinamento básico de modelos.

O foco do curso não é apresentar novos algoritmos, mas mostrar como **decisões de preparação dos dados** impactam diretamente o desempenho de um modelo.

Ao longo do curso, você aprende a:
- lidar com dados incompletos;
- tratar variáveis categóricas;
- construir pipelines consistentes;
- evitar erros comuns em projetos reais.

</details>

---

## 🟩 1.2. Estrutura do Curso

<details>
<br>

O curso é organizado em lições curtas, cada uma focada em um problema comum de Machine Learning:

1. Introduction  
2. Missing Values  
3. Categorical Variables  
4. Pipelines  
5. Cross‑Validation  
6. XGBoost  
7. Data Leakage  

Cada lição apresenta:
- um conceito central;
- exemplos práticos;
- um exercício guiado;
- comparação de abordagens usando uma métrica.

</details>

---

## 🟧 1.3. Dataset Utilizado

<details>
<br>

O curso utiliza o dataset **Ames Housing**, um conjunto de dados tabular com informações sobre imóveis e seus preços de venda.

Características do dataset:
- mistura de variáveis numéricas e categóricas;
- presença de missing values;
- dados realistas, semelhantes a problemas do mundo real.

A variável alvo é:
- **SalePrice** — preço de venda do imóvel.

</details>

---

## 🟨 1.4. Métrica de Avaliação — MAE

<details>
<br>

Todas as decisões do curso são avaliadas usando **MAE (Mean Absolute Error)**.

### O que é MAE
- Média do erro absoluto entre valores reais e previstos.
- Quanto menor o MAE, melhor o modelo.

### Por que MAE é usada
- Fácil de interpretar.
- Penaliza erros de forma linear.
- Adequada para problemas de regressão como o Ames Housing.

~~~python
from sklearn.metrics import mean_absolute_error
~~~

</details>

---

## 🟪 1.5. Modelo Base Utilizado

<details>
<br>

Para comparar abordagens, o curso utiliza o **RandomForestRegressor** como modelo base.

### Por que Random Forest
- bom desempenho em dados tabulares;
- robusto a outliers;
- funciona bem com diferentes tipos de pré‑processamento;
- reduz a influência de escolhas específicas de modelo.

~~~python
from sklearn.ensemble import RandomForestRegressor
~~~

O objetivo não é otimizar o modelo, mas **comparar decisões de preparação dos dados**.

</details>

---

## 🟫 1.6. Metodologia do Curso

<details>
<br>

O padrão seguido em todas as lições é:

1. Identificar um problema comum.
2. Aplicar uma abordagem simples (baseline).
3. Aplicar abordagens alternativas.
4. Comparar resultados usando MAE.
5. Tirar conclusões baseadas em validação.

Esse método reforça uma prática essencial em Machine Learning:
> **não assumir que uma técnica é melhor — medir.**

</details>

---

## 🟦 1.7. Glossário Técnico Inicial

<details>
<br>

- **Dataset** — conjunto de dados usado para treino e validação.
- **Feature** — variável de entrada do modelo.
- **Target** — variável que o modelo tenta prever.
- **Regression** — tipo de problema onde o alvo é numérico.
- **MAE (Mean Absolute Error)** — métrica de erro usada no curso.
- **RandomForestRegressor** — modelo base utilizado para comparação.
- **Validation Set** — conjunto usado para avaliar desempenho fora do treino.

</details>

---

## 🧾 1.8. Referência Rápida — Conceitos‑Chave

<details>
<br>

- **Pré‑processamento importa tanto quanto o modelo.**
- **Compare decisões com métricas.**
- **Use validação para medir impacto real.**
- **Evite conclusões baseadas apenas em intuição.**

Esses princípios guiam todo o curso.

</details>

---

## 🟧 1.9. Conclusão do Capítulo

<details>
<br>

Este capítulo apresentou o contexto, o método e os objetivos do curso.

Nos próximos capítulos, cada conceito será explorado de forma prática, sempre seguindo o mesmo padrão:
- identificar o problema;
- aplicar soluções;
- comparar resultados;
- aprender com os erros.

O próximo capítulo aborda o primeiro desafio real: **Missing Values**.

</details>

</details>
<br>




# 📘 Capítulo 2 — Missing Values

<details>
<br>

> ### *Machine Learning Intermediário — Um Guia Prático e Comentado*

---

## 🟦 2.1. Introdução

<details>
<br>

**Missing Values** são valores ausentes em um dataset e representam um dos problemas mais comuns em dados do mundo real.

No curso *Kaggle Intermediate Machine Learning*, esta lição mostra que:
- modelos não aceitam `NaN` diretamente;
- valores ausentes precisam ser tratados antes do treinamento;
- diferentes estratégias de tratamento produzem resultados diferentes.

O objetivo desta lição é **comparar estratégias de tratamento de missing values usando MAE**, e não apenas aplicar técnicas isoladas.

</details>

---

## 🟩 2.2. Investigação de Missing Values

<details>
<br>

Antes de decidir como tratar missing values, o Kaggle orienta a **investigar onde eles estão**.

### 2.2.1 Quantidade de missing values por coluna

~~~python
X_train.isnull().sum().sort_values(ascending=False).head(10)
~~~

### 2.2.2 Identificação de colunas com missing values

~~~python
cols_with_missing = [
    col for col in X_train.columns
    if X_train[col].isnull().any()
]
cols_with_missing
~~~

### 2.2.3 Por que essa investigação é importante

- Permite entender o impacto de remover colunas.
- Ajuda a decidir se imputação é necessária.
- Indica se a ausência pode carregar informação.

💡 **Orientação do Kaggle:**  
Antes de tratar missing values, identifique onde eles estão e compare abordagens com uma métrica.

</details>

---

## 🟧 2.3. Métrica utilizada no curso — MAE

<details>
<br>

O curso utiliza **MAE (Mean Absolute Error)** para avaliar o impacto das decisões de pré‑processamento.

### O que é MAE
- Média do erro absoluto entre valores reais e previstos.
- Quanto menor o MAE, melhor o modelo.

### Função de avaliação usada no exercício

~~~python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
~~~

💡 **Lição do curso:**  
Decisões de pré‑processamento devem ser comparadas com métricas, não com intuição.

</details>

---

## 🟨 2.4. Strategies for Handling Missing Values

<details>
<br>

O curso apresenta três estratégias simples para lidar com missing values.  
Cada uma modifica o dataset de uma forma diferente, e o impacto é medido com MAE.

---

### 2.4.1 Drop Columns

<details>
<br>

**O que é:**  
Remover colunas que possuem pelo menos um valor ausente.

**O que o código faz:**  
Identifica colunas com `NaN` e remove essas colunas do treino e da validação.

~~~python
cols_with_missing = [
    col for col in X_train.columns
    if X_train[col].isnull().any()
]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
~~~

**O que muda no dataset:**  
- Menos colunas  
- Nenhum valor ausente restante  

**Como aparece no curso:**  
Usado como baseline simples para comparação.

</details>

---

### 2.4.2 Simple Imputation

<details>
<br>

**O que é:**  
Substituir valores ausentes por um valor estatístico (média).

**O que o código faz:**  
Calcula a média no treino e usa esse valor para preencher `NaN` no treino e na validação.

~~~python
from sklearn.impute import SimpleImputer
import pandas as pd

imputer = SimpleImputer(strategy='mean')

imputed_X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

imputed_X_valid = pd.DataFrame(
    imputer.transform(X_valid),
    columns=X_valid.columns
)
~~~

**O que muda no dataset:**  
- Todas as colunas são mantidas  
- `NaN` são substituídos por números  

**Como aparece no curso:**  
Primeira melhoria em relação ao drop.

</details>

---

### 2.4.3 Imputation with Missingness Indicator

<details>
<br>

**O que é:**  
Criar uma coluna extra indicando se o valor original estava ausente.

**O que o código faz:**  
Adiciona colunas booleanas `*_was_missing` antes da imputação.

~~~python
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
~~~

Depois, aplica imputação normalmente.

**O que muda no dataset:**  
- Novas colunas indicam ausência  
- Informação sobre missing é preservada  

**Como aparece no curso:**  
Mostra que a ausência pode ser informativa.

</details>

</details>

---

## 🟫 2.5. Exercício do Kaggle — Início, Execução e Conclusão

<details>
<br>

### 2.5.1 Estratégia 1 — Drop Columns

~~~python
mae_drop = score_dataset(
    reduced_X_train,
    reduced_X_valid,
    y_train,
    y_valid
)
~~~

### 2.5.2 Estratégia 2 — Simple Imputation

~~~python
mae_impute = score_dataset(
    imputed_X_train,
    imputed_X_valid,
    y_train,
    y_valid
)
~~~

### 2.5.3 Estratégia 3 — Imputation + Indicator

~~~python
mae_impute_plus = score_dataset(
    imputed_X_train_plus,
    imputed_X_valid_plus,
    y_train,
    y_valid
)
~~~

### 2.5.4 Comparação Final

~~~python
print("MAE (Drop Columns):", mae_drop)
print("MAE (Simple Imputation):", mae_impute)
print("MAE (Imputation + Indicator):", mae_impute_plus)
~~~

**Conclusão do exercício:**  
A forma como missing values são tratados impacta diretamente o desempenho do modelo.

</details>

---

## 🟩 2.6. Glossário Técnico (Terminologia do Curso)

<details>
<br>

- **Missing Values** — valores ausentes no dataset.  
- **NaN (Not a Number)** — representação de valor ausente.  
- **Imputation** — técnica para preencher valores ausentes.  
- **SimpleImputer** — classe do scikit‑learn para imputação simples.  
- **Missingness Indicator** — coluna extra indicando se o valor estava ausente.  
- **Drop Columns** — remover colunas com valores ausentes.  
- **MAE (Mean Absolute Error)** — erro médio absoluto entre previsão e valor real.  
- **RandomForestRegressor** — modelo baseado em múltiplas árvores de decisão.  

</details>

---

## 🧾 2.7. Referência Rápida — Comandos Importantes

<details>
<br>

- `X.isnull()`  
  Retorna True/False indicando valores ausentes.

- `X.isnull().sum()`  
  Conta missing values por coluna.

- `SimpleImputer(strategy='mean')`  
  Preenche valores ausentes com a média.

- `fit_transform()`  
  Ajusta o imputador e transforma dados de treino.

- `transform()`  
  Aplica o imputador em validation/test.

- `RandomForestRegressor()`  
  Modelo usado no curso para comparação.

- `mean_absolute_error()`  
  Métrica de avaliação do exercício.

</details>

---

## 🟧 2.8. Conclusão do Capítulo

<details>
<br>

Missing values não são apenas um detalhe técnico — são uma etapa fundamental do pré‑processamento.

O curso mostra que:
- investigar vem antes de tratar;
- diferentes estratégias produzem resultados diferentes;
- a escolha correta deve ser guiada por MAE.

Este capítulo estabelece a base para os próximos temas do curso.

</details>

</details>
<br>



# 📘 Capítulo 3 — Categorical Variables

<details>
<br>

> ### *Machine Learning Intermediário — Um Guia Prático e Comentado*

---

## 🟦 3.1. Introdução

<details>
<br>

**Categorical Variables** representam informações qualitativas, como bairros, tipos de rua ou materiais de construção.

Diferente de variáveis numéricas, elas **não possuem relação matemática direta** e, por isso, precisam ser convertidas para números antes de serem usadas em modelos de Machine Learning.

No curso *Kaggle Intermediate Machine Learning*, esta lição mostra que:
- modelos não aceitam strings diretamente;
- diferentes métodos de encoding produzem datasets diferentes;
- a escolha do encoding impacta diretamente o MAE.

</details>

---

## 🟩 3.2. Identificação de Categorical Variables

<details>
<br>

No pandas, variáveis categóricas geralmente aparecem com `dtype == object`.

~~~python
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
object_cols
~~~

Exemplos do dataset Ames Housing:
- `Neighborhood`
- `Condition2`
- `Street`
- `RoofStyle`

Essas colunas precisam ser tratadas antes do treinamento.

</details>

---

## 🟧 3.3. Métrica utilizada no curso — MAE

<details>
<br>

Assim como no capítulo anterior, o curso utiliza **MAE (Mean Absolute Error)** para comparar estratégias de encoding.

~~~python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
~~~

💡 **Lição do curso:**  
A escolha do encoding deve ser guiada por validação e métrica.

</details>

---

## 🟨 3.4. Strategies for Handling Categorical Variables

<details>
<br>

O curso apresenta três estratégias principais para lidar com variáveis categóricas.

Cada estratégia modifica o dataset de uma forma diferente, e o impacto é medido com MAE.

---

### 3.4.1 Drop Categorical Columns

<details>
<br>

**O que é:**  
Remover todas as colunas categóricas.

**O que o código faz:**  
Seleciona apenas colunas numéricas.

~~~python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
~~~

**O que muda no dataset:**  
- Menos colunas  
- Nenhuma variável categórica  

**Como aparece no curso:**  
Usado como baseline simples.

</details>

---

### 3.4.2 Ordinal Encoding

<details>
<br>

**O que é:**  
Converter cada categoria em um número inteiro.

Exemplo:
A → 0
B → 1
C → 2

Código

**Problema apresentado no curso:**  
Algumas categorias aparecem no validation, mas não no treino.

**O que o código faz:**  
Identifica colunas seguras e remove colunas problemáticas.

~~~python
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

good_label_cols = [
    col for col in object_cols
    if set(X_valid[col]).issubset(set(X_train[col]))
]

bad_label_cols = list(set(object_cols) - set(good_label_cols))
~~~

Depois, aplica o encoder apenas nas colunas seguras.

~~~python
from sklearn.preprocessing import OrdinalEncoder

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

encoder = OrdinalEncoder()
label_X_train[good_label_cols] = encoder.fit_transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = encoder.transform(label_X_valid[good_label_cols])
~~~

**O que muda no dataset:**  
- Categorias viram números  
- Algumas colunas podem ser removidas  

**Como aparece no curso:**  
Mostra limitações do Ordinal Encoding.

</details>

---

### 3.4.3 One‑Hot Encoding

<details>
<br>

**O que é:**  
Criar uma coluna binária para cada categoria.

Exemplo:
Neighborhood = CollgCr → [1, 0, 0, ...]
Neighborhood = OldTown → [0, 1, 0, ...]

Código

**O que o código faz:**  
Aplica One‑Hot apenas em colunas de baixa cardinalidade.

~~~python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

OH_cols_train = pd.DataFrame(
    OH_encoder.fit_transform(X_train[low_cardinality_cols])
)
OH_cols_valid = pd.DataFrame(
    OH_encoder.transform(X_valid[low_cardinality_cols])
)

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
~~~

**O que muda no dataset:**  
- Mais colunas  
- Nenhuma ordem artificial  

**Como aparece no curso:**  
Mostra o melhor desempenho quando usado corretamente.

</details>

</details>

---

## 🟪 3.5. Cardinality

<details>
<br>

**Cardinality** é o número de categorias únicas em uma coluna.

~~~python
object_nunique = X_train[object_cols].nunique()
object_nunique.sort_values()
~~~

Exemplos:
| Column | Cardinality |
|------|-------------|
| Street | 2 |
| Condition2 | 8 |
| Neighborhood | 25 |

**Regra prática do curso:**
- Low cardinality (< 10) → One‑Hot Encoding  
- High cardinality (≥ 10) → Drop ou Ordinal  

</details>

---

## 🟫 3.6. Kaggle Exercise — Start, Execution and Conclusion

<details>
<br>

### 3.6.1 Drop Columns

~~~python
mae_drop = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
~~~

### 3.6.2 Ordinal Encoding

~~~python
mae_ordinal = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
~~~

### 3.6.3 One‑Hot Encoding

~~~python
mae_onehot = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)
~~~

### 3.6.4 Final Comparison

~~~python
print("MAE (Drop):", mae_drop)
print("MAE (Ordinal):", mae_ordinal)
print("MAE (One‑Hot):", mae_onehot)
~~~

**Conclusão do exercício:**  
One‑Hot Encoding, quando aplicado corretamente, produz o menor MAE.

</details>

---

## 🟩 3.7. Glossário Técnico

<details>
<br>

- **Categorical Variables** — variáveis que representam categorias.
- **Encoding** — conversão de categorias em números.
- **Ordinal Encoding** — mapeamento de categorias para inteiros.
- **One‑Hot Encoding** — criação de colunas binárias.
- **Cardinality** — número de categorias únicas.
- **MAE** — Mean Absolute Error.
- **RandomForestRegressor** — modelo usado no curso.

</details>

---

## 🧾 3.8. Referência Rápida — Comandos Importantes

<details>
<br>

- `X.select_dtypes(include='object')`
- `OrdinalEncoder()`
- `OneHotEncoder(handle_unknown='ignore')`
- `nunique()`
- `pd.concat()`
- `mean_absolute_error()`

</details>

---

## 🟧 3.9. Conclusão do Capítulo

<details>
<br>

Variáveis categóricas exigem tratamento cuidadoso.  
O curso mostra que:
- diferentes encodings produzem datasets diferentes;
- cardinalidade influencia a escolha do método;
- a decisão correta deve ser validada com MAE.

Este capítulo completa o entendimento de pré‑processamento apresentado no curso.

</details>
</details>
<br>



# 📘 Capítulo 4 — Pipelines

<details>
<br>

> ### *Machine Learning Intermediário — Um Guia Prático e Comentado*

---

## 🟦 4.1. Introdução

<details>
<br>

Nos capítulos anteriores, tratamos **Missing Values** e **Categorical Variables** separadamente.  
Na prática, porém, essas etapas fazem parte de um **único fluxo de pré‑processamento**.

O objetivo deste capítulo é introduzir o conceito de **Pipelines**, que permitem:
- organizar etapas de pré‑processamento;
- garantir consistência entre treino e validação;
- reduzir erros comuns;
- preparar o código para uso em produção.

</details>

---

## 🟩 4.2. O problema sem Pipelines

<details>
<br>

Antes de usar pipelines, o fluxo típico envolve várias etapas manuais:

- tratar missing values;
- aplicar encoding;
- treinar o modelo;
- repetir tudo para o validation set.

Esse processo é:
- repetitivo;
- propenso a erros;
- difícil de manter.

💡 **Problema comum:**  
Esquecer de aplicar exatamente as mesmas transformações no treino e na validação.

</details>

---

## 🟧 4.3. O que é um Pipeline

<details>
<br>

Um **Pipeline** é uma forma de encadear várias etapas de processamento em um único objeto.

No scikit‑learn, um pipeline:
- recebe dados brutos como entrada;
- aplica transformações em sequência;
- entrega dados prontos para o modelo.

Conceitualmente, um pipeline representa:

Raw Data → Preprocessing → Model → Predictions

Código

</details>

---

## 🟨 4.4. Componentes de um Pipeline

<details>
<br>

Um pipeline é composto por **steps**, cada um com um nome e um transformador ou modelo.

Exemplo conceitual:

- imputação de missing values
- encoding de variáveis categóricas
- modelo de regressão

Cada etapa é aplicada automaticamente na ordem definida.

</details>

---

## 🟪 4.5. Por que o curso introduz Pipelines

<details>
<br>

O Kaggle introduz pipelines para resolver problemas vistos nos capítulos anteriores:

- garantir que o mesmo `SimpleImputer` seja usado no treino e na validação;
- evitar vazamento de dados;
- reduzir código duplicado;
- tornar o fluxo mais confiável.

💡 **Mensagem central do curso:**  
Pipelines ajudam a transformar notebooks experimentais em código mais robusto.

</details>

---

## 🟫 4.6. Exemplo conceitual de Pipeline

<details>
<br>

O curso apresenta pipelines como uma forma de unir pré‑processamento e modelo.

Exemplo conceitual (simplificado):

~~~python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', RandomForestRegressor(random_state=0))
])
~~~

Nesse exemplo:
- o imputador trata missing values;
- o modelo recebe dados já tratados;
- o mesmo fluxo é usado no treino e na validação.

</details>

---

## 🟦 4.7. Benefícios práticos dos Pipelines

<details>
<br>

Usar pipelines traz vantagens claras:

- menos código repetido;
- menor chance de erro;
- maior clareza do fluxo;
- facilidade de manutenção;
- preparação para produção.

Esses benefícios se tornam ainda mais importantes conforme o projeto cresce.

</details>

---

## 🟩 4.8. Glossário Técnico

<details>
<br>

- **Pipeline** — encadeamento de etapas de pré‑processamento e modelagem.
- **Step** — etapa individual dentro de um pipeline.
- **Transformer** — objeto que transforma dados (ex.: imputador, encoder).
- **Estimator** — modelo que aprende a partir dos dados.
- **fit()** — ajusta o pipeline aos dados de treino.
- **predict()** — gera previsões usando o pipeline ajustado.

</details>

---

## 🧾 4.9. Referência Rápida — Conceitos‑Chave

<details>
<br>

- Pipelines garantem consistência entre treino e validação.
- Transformações são aplicadas automaticamente na ordem correta.
- O mesmo pipeline pode ser usado em produção.
- Pipelines reduzem risco de data leakage.

</details>

---

## 🟧 4.10. Conclusão do Capítulo

<details>
<br>

Este capítulo introduziu o conceito de **Pipelines** e seu papel no fluxo de Machine Learning.

Nos próximos estudos, o pipeline será usado para:
- integrar imputação e encoding;
- simplificar o código;
- tornar o processo mais confiável.

Após estudar a lição correspondente no Kaggle, este capítulo pode ser expandido com exemplos completos e comparações de MAE.

</details>

</details>


