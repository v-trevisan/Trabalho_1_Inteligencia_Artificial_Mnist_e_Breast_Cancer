# Bases de Dados --- Trabalho de Inteligência Artificial

Este documento apresenta as duas bases de dados escolhidas para o
trabalho prático de análise comparativa de algoritmos de classificação:
**MNIST** e **Breast Cancer Wisconsin (Diagnostic)**.

## 1. MNIST

### 1.1. Descrição

O MNIST (*Modified National Institute of Standards and Technology*) é
uma base de dados utilizada em tarefas de reconhecimento e classificação
de dígitos escritos à mão. Cada exemplo representa uma imagem em escala
de cinza de um algarismo entre **0 e 9**.

A base contém **70.000 imagens**, cada uma com resolução de **28 × 28
pixels**. Portanto, quando cada imagem é representada como um vetor, ela
possui **784 atributos** (28 × 28).

### 1.2. Características

-   **Tipo de problema:** classificação multiclasse.
-   **Quantidade de amostras:** 70.000.
-   **Quantidade de classes:** 10 (dígitos de 0 a 9).
-   **Atributos por amostra:** 784 valores de pixel.
-   **Tipo de atributo:** numérico.
-   **Rótulo:** dígito representado na imagem.

Na versão clássica da base, há 60.000 imagens destinadas ao treinamento
e 10.000 ao teste. Se a base for carregada integralmente pelo OpenML, o
grupo deverá realizar e documentar sua própria divisão entre treinamento
e teste, conforme a metodologia experimental adotada.

### 1.3. Carregamento em Python

A base pode ser obtida pelo `fetch_openml`, disponível no módulo
`sklearn.datasets`:

``` python
from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X_mnist = mnist.data
y_mnist = mnist.target.astype(int)
```

-   `X_mnist` contém os atributos: cada linha representa uma imagem
    convertida em um vetor de 784 pixels.
-   `y_mnist` contém os rótulos corretos, isto é, o dígito representado
    em cada imagem.

Para visualizar uma amostra, é possível reorganizar seu vetor de 784
valores em uma matriz de 28 × 28:

``` python
import matplotlib.pyplot as plt

plt.imshow(X_mnist[0].reshape(28, 28), cmap="gray")
plt.title(f"Classe: {y_mnist[0]}")
plt.axis("off")
plt.show()
```

### 1.4. Relevância para o trabalho

O MNIST permite avaliar o comportamento dos classificadores em um
problema multiclasse com uma quantidade relativamente grande de amostras
e atributos que representam imagens. É interessante observar como a Rede
Neural MLP e o algoritmo de comparação lidam com padrões visuais e com
dez classes diferentes.

**Fonte:** [MNIST --- página de referência de Yann
LeCun](http://yann.lecun.com/exdb/mnist/). A versão `mnist_784` também
está disponível no
[OpenML](https://www.openml.org/search?type=data&sort=runs&id=554).

------------------------------------------------------------------------

## 2. Breast Cancer Wisconsin (Diagnostic)

### 2.1. Descrição

A base **Breast Cancer Wisconsin (Diagnostic)**, também conhecida como
WDBC, reúne características numéricas calculadas a partir de imagens
digitalizadas de amostras de tecido mamário. Cada amostra descreve
propriedades dos núcleos celulares, como raio, textura, perímetro, área
e suavidade.

O objetivo da classificação é distinguir duas categorias: **benigno** e
**maligno**.

A base contém **569 amostras**, descritas por **30 atributos
numéricos**.

### 2.2. Características

-   **Tipo de problema:** classificação binária.
-   **Quantidade de amostras:** 569.
-   **Quantidade de classes:** 2 (benigno e maligno).
-   **Atributos por amostra:** 30.
-   **Tipo de atributo:** numérico.
-   **Rótulo:** diagnóstico benigno ou maligno.

Na versão disponibilizada pelo Scikit-learn, as classes são apresentadas
na seguinte ordem:

  Valor do rótulo   Classe
  ----------------- -----------------------
  `0`               Maligno (`malignant`)
  `1`               Benigno (`benign`)

A base possui 212 amostras malignas e 357 benignas. Essa diferença deve
ser considerada na avaliação: a acurácia, sozinha, pode não representar
adequadamente o desempenho do classificador em cada classe.

### 2.3. Carregamento em Python

A base já está incluída no Scikit-learn e pode ser carregada com
`load_breast_cancer`:

``` python
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

X_bc = breast_cancer.data
y_bc = breast_cancer.target
```

-   `X_bc` contém os 30 atributos numéricos de cada amostra.
-   `y_bc` contém os rótulos de classificação.

Para consultar os nomes das classes:

``` python
print(breast_cancer.target_names)
```

Para consultar os nomes dos atributos:

``` python
print(breast_cancer.feature_names)
```

### 2.4. Relevância para o trabalho

Essa base permite avaliar os classificadores em um problema binário com
atributos numéricos estruturados. Ela também oferece uma oportunidade
para discutir a importância de métricas por classe, especialmente o
*recall* da classe maligna: falsos negativos nessa classe representam
amostras malignas classificadas incorretamente como benignas.

**Fonte:** [UCI Machine Learning Repository --- Breast Cancer Wisconsin
(Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
A base também está disponível na [documentação do
Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

------------------------------------------------------------------------

## 3. Comparação entre as bases

  -----------------------------------------------------------------------
  Característica                          MNIST   Breast Cancer Wisconsin
  ------------------- ------------------------- -------------------------
  Tipo de                           Multiclasse                   Binária
  classificação                                 

  Número de amostras                     70.000                       569

  Número de classes                          10                         2

  Atributos por                             784                        30
  amostra                                       

  Tipo de dado                Pixels de imagens         Medidas numéricas

  Origem dos                 Imagens de dígitos        Imagens de núcleos
  atributos                                                     celulares
  -----------------------------------------------------------------------

A escolha dessas duas bases permite comparar os algoritmos em cenários
distintos. O MNIST apresenta mais amostras, mais atributos e dez
classes, enquanto o Breast Cancer possui menos amostras, atributos
numéricos tabulares e duas classes.

Por isso, os resultados devem ser analisados separadamente em cada base.
A comparação mais importante será entre os algoritmos aplicados ao mesmo
conjunto de dados, utilizando a mesma divisão de treinamento e teste.

## 4. Cuidados metodológicos

Antes do treinamento, alguns cuidados devem ser adotados:

1.  **Separação dos dados:** dividir cada base em conjuntos de
    treinamento e teste. Uma divisão de 80% para treinamento e 20% para
    teste pode ser utilizada, com estratificação (`stratify=y`) para
    preservar aproximadamente a proporção das classes.
2.  **Padronização:** a MLP costuma se beneficiar da padronização dos
    atributos. O ajuste do `StandardScaler` deve ser feito somente com
    os dados de treinamento.
3.  **Ausência de vazamento de dados:** transformações e ajustes de
    hiperparâmetros não devem utilizar informações do conjunto de teste.
4.  **Reprodutibilidade:** definir uma semente aleatória
    (`random_state`) e registrar as configurações utilizadas nos
    experimentos.
5.  **Avaliação por classe:** apresentar matriz de confusão, precisão,
    *recall* e F1-score, além da acurácia. No Breast Cancer, destacar o
    desempenho na classe maligna.
