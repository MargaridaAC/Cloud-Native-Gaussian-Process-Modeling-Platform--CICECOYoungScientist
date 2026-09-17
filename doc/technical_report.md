# Relatório Técnico e Arquitetural: Plataforma Cloud-Native de Modelação por Processos Gaussianos

## 1. Visão Global da Aplicação (End-to-End)

A aplicação é uma plataforma web interactiva e *cloud-native* para modelação probabilística e regressão via **Processos Gaussianos (Gaussian Processes - GP)** com a biblioteca **GPFlow** (baseada em TensorFlow) no backend, e uma interface web SPA (*Single Page Application*) reativa no frontend renderizada com **Plotly.js**.

```
+-----------------------------------------------------------------------------------+
|                                 BROWSER (CLIENT)                                  |
|  [HTML5 / CSS3 / Vanilla JS (app.js)] <---> [Plotly.js Visualizações Interactive]  |
|                     ^ (JSON + X-Session-ID UUID Header)                           |
+---------------------|-------------------------------------------------------------+
                      | HTTP REST API
                      v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                    |
|                                (server.py)                                        |
|  +-----------------------+   +------------------------+   +--------------------+  |
|  | SessionManager        |   | Normalização & Preproc |   | Modelos GPFlow     |  |
|  | (Isolation per UUID)  |   | (Standard, MinMax, etc)|   | (RBF, Matern, etc) |  |
|  +-----------------------+   +------------------------+   +--------------------+  |
+-----------------------------------------------------------------------------------+
                      |                                        |
                      v                                        v
        +---------------------------+            +----------------------------+
        |  SESSÕES ISOLADAS         |            |  HEALTH CHECKS / CONTAINER |
        |  (/tmp/sessions/ & Redis) |            |  (/healthz & /ready)       |
        +---------------------------+            +----------------------------+
```

### O Fluxo Completo de Ponta a Ponta:
1. **Carregamento Inicial**: O browser solicita a raiz `/`, recebendo [`index.html`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/index.html), [`styles.css`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/css/styles.css) e [`app.js`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/js/app.js).
2. **Identificação Única de Sessão**: O frontend inicializa a sessão gerando um UUID v4 único (guardado no `sessionStorage`), anexado a todos os pedidos através do cabeçalho HTTP `X-Session-ID`.
3. **Entrada de Dados**: O utilizador insere pontos manualmente ou carrega um ficheiro CSV contendo variáveis independentes (features) e a variável dependente (target/label).
4. **Treino do Modelo GP**: Ao submeter a configuração (Kernel RBF/Matern, normalização, likelihood noise, split de treino/teste), o FastAPI aciona a função `fit_gp_model()`. O GPFlow calcula o hiperparâmetros ótimos minimizando a perda de treino (*Negative Log Marginal Likelihood*) com o otimizador Scipy (`L-BFGS-B`).
5. **Previsão e Visualização**: O backend calcula as previsões médias \(\mu(X)\) e as variâncias/desvios-padrão \(\sigma^2(X)\), devolvendo matrizes JSON. O Plotly.js desenha curvas 2D com bandas de confiança sombreadas (RGBA) ou superfícies 3D (`surface`).

---

## 2. Arquitetura e Ciclo de Vida de um Pedido

### A. Geração e Envio de Session ID no Frontend (`app.js`)
No arranque da aplicação no browser, a função `getSessionId()` é executada:
```javascript
function getSessionId() {
    let sid = sessionStorage.getItem("gp_session_id");
    if (!sid) {
        sid = crypto.randomUUID();
        sessionStorage.setItem("gp_session_id", sid);
    }
    return sid;
}
```
Todos os pedidos HTTP enviados ao backend utilizam a função wrapper `apiFetch()`:
```javascript
async function apiFetch(url, options = {}) {
    options.headers = options.headers || {};
    options.headers["X-Session-ID"] = getSessionId();
    return fetch(url, options);
}
```

### B. Isolamento e Gestão no Backend (`server.py`)
No backend, a classe `SessionManager` extrai o cabeçalho `X-Session-ID`. 
* Cada pedido obtém o seu estado isolado recuperando-o do armazenamento persistente em disco (`/tmp/sessions/{session_id}.json`) ou do servidor **Redis** (se a variável de ambiente `REDIS_URL` estiver configurada).
* Os objetos de modelo `gpflow.models.GPR` ativos são mantidos no dicionário em memória `SESSION_MODELS[session_id]`.
* Uma rotina assíncrona limpa periodicamente sessões inativas há mais de 1 hora.

### C. Exemplo Prático: Fluxo de Treino (`/api/train-manual`) e Predição (`/api/predict-y`)

```
 [UTILIZADOR] -> Insere X=[1,2,3,4,5], Y=[1.5,3.1,4.8,6.9,9.2] + Clica "Train Model"
      |
      v
  [app.js] -> apiFetch("/api/train-manual", { body: JSON, headers: {"X-Session-ID": "uuid-123"} })
      |
      v
 [server.py /api/train-manual]:
   1. SessionManager.get_session("uuid-123") -> Obtém estado limpo ou existente.
   2. Normalization() -> Transforma X e Y conforme a opção escolhida (Standardization / MinMax / None).
   3. fit_gp_model() -> Cria gpflow.models.GPR((X_N, Y_N), kernel=SquaredExponential()).
   4. Scipy.minimize() -> Ajusta os parâmetros do kernel.
   5. SessionManager.save_session("uuid-123") -> Persiste o estado atualizado.
      |
      v
  [app.js] -> Notifica sucesso e atualiza indicadores de estado da UI.
      |
      v
 [UTILIZADOR] -> Pede predição no ponto X=2.5 com Nível de Confiança=95%
      |
      v
  [app.js] -> apiFetch("/api/predict-y", { body: {"x_values": [2.5], "confidence_level": 95} })
      |
      v
 [server.py /api/predict-y]:
   1. Recupera o modelo ajustado da sessão "uuid-123".
   2. model.predict_y(x_val_N) -> Retorna (Y_mean_N, Y_var_N).
   3. Des-normaliza média e variância de volta para a escala original dos dados.
   4. Calcula o score z da distribuição normal std (ex: z=1.96 para 95% CI).
   5. Retorna {"pred_y": 3.98, "std_y": 0.12, "ci_lower": 3.74, "ci_upper": 4.22}.
```

---

## 3. Mecanismos de Robustez e Cloud-Native

### A. Eliminação do `SESSION_STATE` Global
* **O Perigo Anterior**: Em versões monolíticas antigas, o estado da aplicação era guardado numa única variável global Python (`SESSION_STATE = {}`). Em ambientes multi-utilizador ou em contentores concorrentes, se dois utilizadores treinassem modelos em simultâneo, o Utilizador B sobrescreveria os dados e o modelo do Utilizador A, resultando em **vazamento de dados (cross-user data leakage)** e respostas inconsistentes.
* **A Solução Implementada**: Eliminação total de variáveis globais mutáveis de estado. O estado é totalmente desacoplado da instância do processo FastAPI e isolado por `session_id`.

### B. Health Checks para Orquestração (`/healthz` e `/ready`)
Em orquestradores de contentores modernos (Kubernetes, AWS ECS, Docker Swarm):
* **`/healthz` (Liveness Probe)**: Indica se a aplicação Python/FastAPI está viva e a responder a pedidos HTTP. Se este endpoint falhar, o orquestrador reinicia o contentor (Restart Policy).
* **`/ready` (Readiness Probe)**: Indica se o serviço está pronto para receber tráfego de produção (ex: dependências inicializadas, armazenamento acessível). Se falhar, o orquestrador remove o contentor da rota do Load Balancer.

### C. Segurança e Estabilidade no Dockerfile
* **Utilizador Não-Root (`appuser`)**:
  * *Razão*: Executar aplicações como `root` num contentor Docker abre vulnerabilidades críticas de elevação de privilégios e *container escape*. Ao criar o utilizador com ID fixo (`useradd -u 1000 appuser`), garantimos o princípio do menor privilégio (*Least Privilege*).
* **Variáveis de Ambiente (`PORT`, `HOST`, `LOG_LEVEL`)**:
  * *Razão*: Segue os princípios da *12-Factor App*. Permite alterar portas ou níveis de log no arranque do contentor sem recalcular a imagem Docker.
* **Diretiva `HEALTHCHECK`**:
  * *Razão*: Permite ao Docker daemon monitorizar autonomamente a saúde interna da aplicação via `curl` a cada 30 segundos.

---

## 4. Testes e Automação (CI/CD)

### A. Funcionamento do Pipeline GitHub Actions (`.github/workflows/ci.yml`)
O pipeline executa automaticamente a cada `push` ou `pull_request` para a branch `main`:

```
 [PUSH / PR para main]
          |
          v
   +-------------------------------------------------------------+
   | Job 1: Lint & Python Validation (Ubuntu Latest, Python 3.12)|
   |  1. Checkout do código-fonte                               |
   |  2. Instalação de dependências (requirements.txt + ruff)    |
   |  3. Linting e análise estática: ruff check .               |
   |  4. Execução da suite de testes: pytest                    |
   +-------------------------------------------------------------+
          | (Apenas se Job 1 passar com Sucesso)
          v
   +-------------------------------------------------------------+
   | Job 2: Docker Build Test                                   |
   |  1. Configuração do Docker Buildx                           |
   |  2. Teste de compilação da imagem Dockerfile               |
   +-------------------------------------------------------------+
```

### B. Função da Suite de Testes Unitários (`tests/test_api.py`)
Os testes criados garantem regressão zero e validam cenários críticos antes do deploy:
1. `test_healthz` & `test_ready`: Garantem que os endpoints de orquestração respondem `200 OK` e `{"status": "ok"}`.
2. `test_session_isolation`: Simula dois utilizadores concorrentes enviando pedidos simultâneos com cabeçalhos `X-Session-ID` distintos. Garante formalmente que a aprendizagem e estado do Utilizador 1 não afetam o Utilizador 2.
3. `test_train_and_predict`: Executa um fluxo completo de ponta a ponta (treino de modelo GP 1D e predição com intervalo de confiança) confirmando que a matemática do GPFlow e a desserialização Pydantic funcionam de forma determinística.

---

## 5. Mapa Físico dos Ficheiros do Projeto

| Ficheiro / Caminho | Papel no Sistema |
| :--- | :--- |
| [`server.py`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/server.py) | **Backend REST API (FastAPI)**: Gere a lógica de negócio, ajustamento de modelos GPFlow, serialização JSON Pydantic, isolamento de sessões e endpoints HTTP (`/api/*`, `/healthz`, `/ready`). |
| [`static/js/app.js`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/js/app.js) | **Lógica Frontend (SPA)**: Gere a interação com a UI, injeção automática de `X-Session-ID`, chamadas AJAX ao backend e renderização de gráficos 2D/3D no Plotly.js. |
| [`static/index.html`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/index.html) | **Interface do Utilizador**: Estrutura HTML5 com barra lateral de navegação, modais de parâmetros, tabelas de dados e contentores de gráficos Plotly. |
| [`static/css/styles.css`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/static/css/styles.css) | **Estilos Visuais**: CSS personalizado com tema moderno, modo escuro/claro, sombras, transições e regras de responsividade. |
| [`tests/test_api.py`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/tests/test_api.py) | **Suite de Testes (pytest)**: Testes automatizados para validação de endpoints HTTP, isolamento de sessões multi-utilizador e cálculo de predições. |
| [`Dockerfile`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/Dockerfile) | **Especificação de Containerização**: Imagem `python:3.12-slim`, utilizador não-root `appuser`, HEALTHCHECK ativo e exposição da porta 7860. |
| [`requirements.txt`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/requirements.txt) | **Gestão de Dependências**: Lista declarativa das bibliotecas de produção e desenvolvimento (`fastapi`, `gpflow`, `tensorflow`, `pytest`, `httpx`, etc.). |
| [`.github/workflows/ci.yml`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/.github/workflows/ci.yml) | **Pipeline CI/CD**: Automação do GitHub Actions para verificação sintática com `ruff`, execução da suite `pytest` e teste de build da imagem Docker. |
| [`doc/how-to-run.md`](file:///c:/Users/almei/Desktop/Cloud-Native-Gaussian-Process-Modeling-Platform--CICECOYoungScientist/doc/how-to-run.md) | **Documentação Operacional**: Guia passo-a-passo para arranque local com venv Python e execução em contentor Docker. |
