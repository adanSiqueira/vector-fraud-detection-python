  [🇺🇸 English](README.md) | [🇧🇷 Português](README.pt-BR.md) 


<div align="center">

# Rinha de Backend 2026 — API de Detecção de Fraudes - Python

<p align="center"> 
  <img src="https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" /> 
  <img src="https://img.shields.io/badge/starlette-0.41.3-black?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/granian-2.7.4-orange?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/nginx-1.27-green?style=for-the-badge&logo=nginx&logoColor=white" /> 
  <img src="https://img.shields.io/badge/faiss-ivf+sq8-red?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/orjson-rust%20powered-purple?style=for-the-badge" /> 
  <img src="https://img.shields.io/badge/docker-compose-blue?style=for-the-badge&logo=docker&logoColor=white" />
</p>

</div>

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): uma API de detecção de fraudes utilizando busca vetorial por vizinhos aproximados sobre 3 milhões de transações rotuladas.

---

## Por que Python?

Este projeto foi intencionalmente desenvolvido em Python, mesmo sabendo que a competição é fortemente dominada por linguagens de mais baixo nível como Rust, Go e C, que naturalmente possuem vantagens em throughput bruto e latência.

O objetivo aqui não é necessariamente superar esses ecossistemas, mas **explorar até onde o Python pode ser levado quando cada milissegundo importa**.

O projeto combina decisões arquiteturais de baixo overhead, dependências com runtime em Rust, otimizações conscientes de memória, técnicas de compressão de busca vetorial e ajuste cuidadoso do caminho crítico para tornar o Python tão performático quanto possível dentro das restrições da competição.

---

## O desafio real: memória, não CPU

No início, **tentei utilizar `hnswlib` com um índice HNSW**.

O problema foi descoberto apenas após múltiplos testes de stress locais e execuções completas de orquestração Docker: o consumo de memória era fundamentalmente incompatível com os limites da competição.

As regras da competição exigem:

- pelo menos 2 instâncias de API
- 1 balanceador de carga
- orçamento total máximo de:
  - 1 CPU
  - 350 MB RAM

**A abordagem original com HNSW se tornou inviável porque o HNSW armazena toda a estrutura de grafo em memória para cada vetor.**

Com:

- 3 milhões de vetores
- 14 dimensões
- HNSW `M=8`

o índice consumia aproximadamente:

```
~721 MB RAM
```

Nenhum ajuste de parâmetros conseguiria fazer dois contêineres de API coexistir dentro do orçamento de memória permitido.

O gargalo não era o Python em si.

Era o modelo de memória do índice vetorial.

---

## A solução: Faiss IVF + SQ8

**Faiss (Facebook AI Similarity Search)** é uma biblioteca de busca por similaridade vetorial de alto desempenho desenvolvida pela Meta AI, amplamente utilizada em sistemas de recomendação, busca semântica, recuperação de embeddings e cargas de trabalho de vizinhos mais próximos em grande escala.

Para este projeto, o tipo de índice escolhido foi:

```python
faiss.IndexIVFScalarQuantizer(...)
````

que combina duas técnicas complementares:

* IVF (Inverted File Index — Índice de Arquivo Invertido)
* SQ8 (Quantização Escalar de 8 bits)

O motivo desta escolha foi a eficiência de memória.

Embora o HNSW ofereça excelente recall e latência, sua estrutura baseada em grafos se torna extremamente intensiva em memória em grande escala, pois toda a conectividade do grafo precisa permanecer residente na RAM.

O Faiss IVF+SQ8 troca uma pequena quantidade de recall por uma redução massiva no uso de memória, mantendo ainda tempos de consulta abaixo de um milissegundo.

Isso tornou possível encaixar 3 milhões de vetores dentro dos rígidos limites de memória dos contêineres da competição.

A arquitetura foi redesenhada em torno de duas técnicas complementares do Faiss:

### 1. IVF — Inverted File Index (Índice de Arquivo Invertido)

O conjunto de dados é particionado em 1000 clusters durante a construção.

No momento da consulta, apenas 10 clusters são pesquisados (`nprobe=10`), reduzindo o espaço de busca para aproximadamente 1% do conjunto completo.

Em vez de:

**O(N)**

as consultas se tornam aproximadamente:

**O(N / nlist * nprobe)**

Isso reduz drasticamente o custo de busca enquanto mantém um bom recall.

---

### 2. SQ8 — Quantização Escalar (8 bits)

Cada dimensão float32 é comprimida:

**float32 (4 bytes) → uint8 (1 byte)**

Isso proporciona:

* ~4× de redução de memória
* degradação mínima de precisão
* ~97,4% de recall medido em comparação com IVFFlat

Como os vetores já estão normalizados em `[0,1]`, a quantização escalar funciona extremamente bem para este conjunto de dados.

---

## Impacto na memória

A migração de HNSW para Faiss IVF+SQ8 mudou completamente a viabilidade da arquitetura.

| Componente          | RAM     |
| ------------------- | ------- |
| Índice Faiss IVF+SQ8 | ~64 MB  |
| labels.npy          | ~3 MB   |
| Python + granian    | ~50 MB  |
| Total por contêiner | ~117 MB |

Comparado com a abordagem original HNSW:

| Tipo de índice | RAM aproximada |
| -------------- | -------------- |
| hnswlib HNSW   | ~721 MB        |
| Faiss IVF+SQ8  | ~64 MB         |

Essa redução tornou as restrições da competição alcançáveis.

---

## Arquitetura

```
Client
  │  POST /fraud-score  (porta 9999)
  ▼
┌──────────────────────────────────┐
│ nginx 1.27-alpine                │
│ Balanceador de carga round-robin │
│ 0.20 CPU · 30 MB                 │
└────────────┬─────────────────────┘
             │ round-robin
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐     ┌─────────┐
│ api1    │     │ api2    │
│ granian │     │ granian │
│ 1 worker│     │ 1 worker│
│ 0.50CPU │     │ 0.40CPU │
│ 280 MB  │     │ 160 MB  │
└─────────┘     └─────────┘

Índice Faiss IVF+SQ8 pré-construído durante o docker build
```


Total de recursos declarados: **1.00 CPU · 350 MB RAM**


Totalmente em conformidade com as regras da competição.

---

## Problema de orquestração na inicialização

Após resolver o problema de consumo de memória, outro problema surgiu:

`api1` iniciava corretamente, mas `api2` frequentemente falhava ao carregar o mesmo índice simultaneamente.

A causa raiz era a pressão de memória durante o carregamento simultâneo do índice.

Mesmo que o uso de RAM em estado estável coubesse confortavelmente dentro do limite, o Docker experimentava brevemente um pico de memória enquanto ambos os contêineres carregavam sua própria cópia do índice ao mesmo tempo.

O sistema operacional ainda mantinha as páginas do índice do primeiro contêiner aquecidas na memória enquanto o segundo contêiner começava a ler o mesmo arquivo.

---

## A correção: sequenciamento de inicialização com atraso

A solução foi intencionalmente simples e determinística:

`api2` aguarda antes de iniciar.

```yaml
command:
  [
    "sh",
    "-c",
    "sleep 15 && python -m granian ..."
  ]
```

Isso permite:

* que a api1 inicialize completamente
* que o cache de páginas do SO se estabilize
* que a pressão de I/O caia
* que os picos de memória desapareçam

Após introduzir o sequenciamento de inicialização, ambos os contêineres passaram a operar de forma estável simultaneamente dentro dos limites da competição.

---

## Decisões de performance

### 1. Starlette em vez de FastAPI

O FastAPI adiciona camadas de injeção de dependência e validação que custam latência mensurável por requisição.

Usar Starlette puro remove esse overhead enquanto preserva a ergonomia ASGI.

---

### 2. orjson em vez de json da stdlib

`orjson` é baseado em Rust e significativamente mais rápido que a implementação JSON padrão do Python tanto para serialização quanto para desserialização.

---

### 3. Sem Pydantic

A competição garante payloads válidos.

Evitar a validação de schema remove alocações e overhead de CPU desnecessários.

Todos os campos são acessados diretamente de dicionários brutos.

---

### 4. Buffers NumPy pré-alocados por thread

Cada worker aloca um único buffer:

```python
(1, 14) float32
```

e o reutiliza para cada requisição.

Benefícios:

* zero alocações NumPy por requisição
* memória contígua amigável ao cache
* layout exato esperado pelo Faiss

---

### 5. granian em vez de uvicorn

`granian` utiliza um runtime Rust e consistentemente apresenta melhor throughput e latência p99 do que servidores ASGI em Python puro.

---

### 6. Geração do índice em tempo de build

O índice Faiss é construído durante o `docker build`.

Os contêineres realizam apenas:

```python
faiss.read_index(...)
```

durante a inicialização.

Benefícios:

* sem custo de treinamento em runtime
* inicialização determinística
* evita picos transientes massivos de RAM

---

### 7. Imagem Docker multi-stage

A imagem de runtime contém:

* nenhum compilador
* nenhum build-essential
* nenhuma etapa de pip install

Apenas pacotes Python pré-construídos e o índice Faiss gerado são copiados do estágio de build.

Isso reduz:

* tamanho da imagem
* complexidade de inicialização
* dependências em runtime

---

### 8. nginx ajustado para baixa latência

O balanceador de carga é configurado como um proxy de passagem puro com:

* keepalive upstream
* `tcp_nodelay`
* buffering desabilitado
* logs de acesso desabilitados

As regras proíbem explicitamente lógica de negócio no balanceador de carga.

---

## Stack

| Componente       | Escolha           | Por quê                              |
| ---------------- | ----------------- | ------------------------------------ |
| Servidor ASGI    | granian 2.7.4     | Runtime Rust, menor latência p99     |
| Framework web    | Starlette 0.41.3  | Overhead mínimo                      |
| JSON             | orjson 3.11.9     | JSON baseado em Rust                 |
| Busca vetorial   | Faiss IVF+SQ8     | Redução massiva de RAM               |
| Numérico         | numpy 2.4.4       | Matemática vetorizada                |
| Balanceador      | nginx 1.27-alpine | Proxy reverso leve                   |
| Containerização  | Docker Compose    | Orquestração da competição           |

---

## Árvore de arquivos

```
.
├── app/
│   ├── main.py
│   └── requirements.txt
├── scripts/
│   └── build_index.py
├── resources/
│   └── references.json.gz
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── test_vectorize.py
├── README.md
├── README.pt-BR.md
└── .gitignore
```

---

## Endpoints

| Método | Caminho        | Descrição                               |
| ------ | -------------- | --------------------------------------- |
| GET    | `/ready`       | Verificação de saúde                    |
| POST   | `/fraud-score` | Retorna decisão e pontuação de fraude   |
