# LLM4TSF Leaderboard - Streamlit

Leaderboard em Streamlit para comparar LLMs em previsão de séries temporais, inspirado no estilo de leaderboards públicos como os do Hugging Face: ranking transparente, filtros, gráficos, tabelas auditáveis e fluxo de submissão.

## Estrutura

```text
llm_tsf_leaderboard/
├── app.py
├── leaderboard_app/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   └── ui.py
├── requirements.txt
└── data/
    ├── results.csv
    ├── sample_prediction_submission.csv
    └── sample_results_submission.csv
```

## Como rodar

```bash
cd llm_tsf_leaderboard
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
streamlit run app.py
```

## Páginas do app

1. **Ranking principal**: rank geral dos modelos, filtros por estratégia de prompt e gráficos.
2. **Resultados completos**: tabela bruta, pivôs, diagnóstico de cobertura e exportação.
3. **Guia de cálculo**: demonstração editável de SMAPE, MAE, RMSE e decomposição do score geral.
4. **Demo & submissão**: cálculo manual de métricas e upload de CSV para novas submissões.
5. **Notebook exemplo**: link para o notebook `Carbon_Qwen3_8b.ipynb` e checklist de exportação.

## Esquema mínimo do CSV principal

O arquivo `data/results.csv` pode ser substituído pelos seus resultados reais. As colunas principais são:

- `model_name`
- `model_id`
- `family`
- `provider`
- `deployment`: `Local`, `API`, `Hybrid`
- `dataset`
- `horizon`
- `prompt_strategy`: `zero-shot`, `few-shot`, `cot`, `cot+few`
- `smape`
- `mae`
- `rmse`
- `inference_time_s`
- `co2_g`
- `cost_usd_per_1k_forecasts`
- `run_id`
- `evaluation_date`
- `hardware`
- `quantization`

Colunas adicionais recomendadas:

- `execution_source`: por onde o modelo rodou, por exemplo `Local`, `OpenAI API`, `NVIDIA NIM API`.
- `energy_kwh_per_1k_forecasts`: energia por 1.000 previsões, quando houver medição local.
- `api_pricing_source`: fonte do preço da API.
- `notes`: observações sobre execução, limitações e telemetria.

## Upload de CSV na página de submissão

A página **Demo & submissão** aceita dois formatos.

### Formato A — resultados já calculados

Use quando o notebook/script já calculou SMAPE, MAE e RMSE.

```csv
model_name,model_id,family,provider,deployment,dataset,horizon,prompt_strategy,smape,mae,rmse,inference_time_s,co2_g,cost_usd_per_1k_forecasts,run_id,evaluation_date
Qwen3-8B,Qwen/Qwen3-8B,Qwen,Local,Local,Carbon,96,zero-shot,12.4,0.084,0.115,1.38,0.91,0.0000,1,2026-05-28
```

### Formato B — previsões brutas

Use quando você tem `y_true` e `y_pred`. O app calcula SMAPE, MAE e RMSE e agrega por modelo/dataset/horizonte/estratégia/run.

```csv
model_name,model_id,dataset,horizon,prompt_strategy,run_id,y_true,y_pred
Qwen3-8B,Qwen/Qwen3-8B,Carbon,96,zero-shot,1,10.2,10.0
Qwen3-8B,Qwen/Qwen3-8B,Carbon,96,zero-shot,1,11.0,10.9
```

Também são aceitas células com listas, por exemplo:

```csv
model_name,model_id,dataset,horizon,prompt_strategy,run_id,y_true,y_pred
Qwen3-8B,Qwen/Qwen3-8B,Carbon,96,zero-shot,1,"[10.2,11.0,10.8]","[10.0,10.9,11.1]"
```

## Regra para submissões locais e via API

O app exige que o usuário indique **por onde rodou**:

- Se for **local**, informe hardware, quantização e, se disponível, energia por 1.000 previsões.
- Se for **API**, informe custo financeiro por 1.000 previsões e fonte do preço.
- Para API sem telemetria confiável de energia/carbono, registre a ausência em `notes` e escolha uma política de ranking coerente na barra lateral.

## Fórmula do ranking

Todas as métricas são do tipo “quanto menor, melhor”. Para cada métrica, o app aplica normalização min-max invertida:

```text
score_metrica = 100 * (max(valor) - valor) / (max(valor) - min(valor))
```

O score final é uma média ponderada dos scores normalizados. Por padrão:

- SMAPE: 35%
- MAE: 15%
- RMSE: 15%
- Tempo de inferência: 15%
- CO₂: 10%
- Custo financeiro: 10%

Você pode alterar os pesos pela barra lateral.

## Notebook de exemplo

O app inclui uma página com o link para:

```text
https://github.com/llm4time/NeurIPS2026/blob/main/Carbon_Qwen3_8b.ipynb
```

Use o notebook como referência para executar o modelo, medir tempo/energia/carbono e exportar um CSV compatível com a página de submissão.

## Observação

Os dados incluídos em `data/results.csv` são sintéticos e servem apenas para testar a interface. Substitua pelos resultados reais do benchmark antes de publicar.
