# Relatório Técnico — Processo de Fine-Tuning

Autor: Jeferson Furtado
Projeto: Médico IA LLM (`medllm_healthqa_br_colab.ipynb`)

Este documento detalha o processo de fine-tuning aplicado no projeto, atendendo ao requisito de relatório técnico do trabalho.

## 1. Visão geral

O objetivo foi adaptar um modelo de linguagem geral (Llama-3-8B) para responder questões médicas em português, a partir de um dataset de provas de residência/revalidação médica brasileiras. A técnica escolhida foi **QLoRA** (Quantized Low-Rank Adaptation), executada com a biblioteca **Unsloth**, que otimiza o processo para caber e treinar em uma única GPU T4 (disponível gratuitamente no Google Colab).

Fine-tuning completo (atualizando todos os ~8 bilhões de parâmetros do modelo) exigiria dezenas de GB de VRAM e não é viável em uma T4 (16 GB). QLoRA resolve isso combinando duas técnicas:

1. **Quantização 4-bit**: os pesos do modelo base são carregados comprimidos em 4 bits (em vez dos 16/32 bits originais), reduzindo drasticamente o uso de memória.
2. **LoRA (Low-Rank Adaptation)**: em vez de atualizar os pesos originais, pequenas matrizes de baixo posto ("adaptadores") são inseridas em camadas específicas do modelo e são as únicas partes efetivamente treinadas. O modelo base permanece congelado.

## 2. Modelo base

- **Checkpoint**: `unsloth/llama-3-8b-bnb-4bit` (Llama 3, 8 bilhões de parâmetros, checkpoint **base**, não `-Instruct`, já pré-quantizado em 4-bit pela Unsloth).
- **Alternativa disponível no notebook**: `unsloth/mistral-7b-bnb-4bit` (não utilizada na versão final).
- **Comprimento máximo de sequência**: 1024 tokens (`MAX_SEQ_LENGTH`).
- Carregado via `FastLanguageModel.from_pretrained(...)`, que já traz o modelo quantizado e otimizado pela Unsloth (kernels mais rápidos, menor uso de memória que a implementação padrão do `transformers` + `bitsandbytes`).

Por ser um checkpoint **base** (não instruído), o modelo não segue instruções por padrão — é justamente o fine-tuning supervisionado (SFT) que ensina o padrão de pergunta/resposta usado neste projeto.

## 3. Configuração do LoRA

Aplicado via `FastLanguageModel.get_peft_model(...)`, com os seguintes parâmetros:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `r` (rank) | 16 | Dimensão das matrizes de baixo posto — controla a capacidade do adaptador |
| `lora_alpha` | 16 | Fator de escala aplicado às atualizações do LoRA |
| `lora_dropout` | 0.0 | Sem dropout nos adaptadores |
| `bias` | `"none"` | Nenhum bias adicional treinado |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Todas as projeções de atenção e da MLP recebem adaptadores — cobertura completa do bloco transformer |
| `use_gradient_checkpointing` | `"unsloth"` | Checkpointing de gradiente otimizado pela Unsloth, reduz uso de VRAM durante o backward |
| `random_state` | 3407 (`SEED`) | Garante reprodutibilidade da inicialização dos adaptadores |

Com essa configuração, apenas uma fração pequena dos parâmetros totais do modelo é efetivamente treinada (tipicamente < 1% do total, como observado no notebook original em inglês, que reportou 0,52% dos parâmetros treináveis com configuração equivalente).

## 4. Dataset e preparação dos dados

- **Fonte**: [`Larxel/healthqa-br`](https://huggingface.co/datasets/Larxel/healthqa-br), carregado via `load_dataset("Larxel/healthqa-br", split="train")`.
- **Formato original**: questões de múltipla escolha de provas de residência/revalidação médica no Brasil (ex.: Revalida), com os campos:
  - `question`: enunciado do caso clínico, com as alternativas A-E embutidas no próprio texto;
  - `answer`: letra da alternativa correta;
  - `source`: origem da questão (ex.: "Revalida");
  - `year`: ano da prova.
- O dataset **não** possui um campo de explicação/justificativa pronto — apenas a letra correta. A resposta de treino, portanto, é **construída programaticamente** a partir desses campos (função `format_example`), e não copiada diretamente do dataset.
- O dataset é público e não contém dados de pacientes reais (são questões de provas médicas). Um exemplo de registros pode ser visto em [`exemplo_dataset_healthqa_br.csv`](exemplo_dataset_healthqa_br.csv).

### 4.1 Duplo formato: múltipla escolha + pergunta aberta

Para evitar que o modelo aprendesse a responder **apenas** no formato de múltipla escolha (um risco real de fine-tuning em um único formato rígido — conhecido como "format collapse"), o processo de formatação divide os exemplos em dois grupos, sorteados com uma semente fixa (`_rng = random.Random(SEED)`) para reprodutibilidade:

- **~60% dos exemplos — múltipla escolha** (formato original): o enunciado mantém as alternativas A-E, e a resposta de treino começa com `"Alternativa correta: {letra} — {texto da alternativa}"`.
- **~40% dos exemplos — pergunta aberta** (`PROPORCAO_PERGUNTA_ABERTA = 0.4`): a função `remover_alternativas()` extrai apenas o caso clínico e a pergunta, descartando o bloco de alternativas via expressão regular. A resposta de treino, nesse caso, é o texto livre da alternativa correta (sem o prefixo "Alternativa correta: X"), extraído com `extrair_texto_alternativa()`.

Em **ambos os casos**, a resposta de treino sempre termina com:
1. A citação da fonte: `"Fonte: {source} ({year})."`;
2. Uma frase fixa de recomendação de acompanhamento médico (`RECOMENDACAO_MEDICA`).

Isso faz com que esses dois elementos — explicabilidade (fonte) e limite de atuação (recomendação médica, nunca uma prescrição definitiva) — sejam aprendidos pelo modelo como parte do **padrão de resposta**, e não apenas exigidos via instrução de prompt em tempo de inferência.

### 4.2 Template de prompt (padrão Alpaca)

```
Abaixo está uma questão médica. Responda com base no caso apresentado — se houver
alternativas, indique a correta; se não houver, responda em texto livre. Cite a fonte
da questão e finalize sempre recomendando acompanhamento médico.

### Questão:
{questão}

### Resposta:
{resposta}
```

O uso de um único template genérico (em vez de um texto específico para "múltipla escolha") é o que permite treinar os dois formatos com a mesma estrutura de prompt, deixando o modelo aprender a se adaptar ao conteúdo da pergunta.

Cada exemplo formatado recebe o token de fim de sequência (`eos_token`) do tokenizer ao final, para marcar onde a resposta termina durante o treino.

## 5. Processo de treinamento (SFT)

O treinamento supervisionado é feito com `SFTTrainer` da biblioteca **TRL** (Transformer Reinforcement Learning), que implementa fine-tuning supervisionado padrão sobre um campo de texto já formatado (`dataset_text_field="text"`).

### 5.1 Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| `max_steps` | 60 (controle por passos, não por épocas) |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 |
| Batch efetivo | 8 (2 × 4) |
| `learning_rate` | 2e-4 |
| `lr_scheduler_type` | linear |
| `warmup_steps` | 5 |
| `weight_decay` | 0.01 |
| `optim` | `adamw_8bit` (otimizador Adam com estados em 8-bit, economiza VRAM) |
| Precisão | `fp16` (ou `bf16` se a GPU suportar — a T4 não suporta `bf16`, então usa `fp16`) |
| `packing` | `False` (cada exemplo é tratado individualmente, sem concatenar sequências) |
| `seed` | 3407 |

O treino é controlado por **número de passos** (`max_steps=60`), não por épocas — uma escolha deliberada para tornar o tempo de treino previsível (ordem de 15-25 minutos em uma T4), independentemente do tamanho do dataset.

### 5.2 Por que essa combinação de técnicas

- **Unsloth**: reescreve os kernels de atenção e do LoRA em Triton, tornando o treino ~2x mais rápido e reduzindo o uso de memória em relação à combinação padrão `transformers` + `peft` + `bitsandbytes`, sem alterar o resultado matemático do treino.
- **Gradient checkpointing**: troca computação por memória — em vez de guardar todas as ativações intermediárias para o backward, elas são recalculadas quando necessário. Essencial para caber um modelo de 8B em uma T4 (16 GB).
- **Otimizador 8-bit (`adamw_8bit`)**: os estados do otimizador Adam (momentos de 1ª e 2ª ordem) são armazenados em 8 bits em vez de 32, economizando VRAM adicional sem alterar significativamente a qualidade da convergência.
- **`report_to="none"`**: desativa a integração com Weights & Biases, evitando que o notebook trave esperando um login interativo.

## 6. Pós-treino: salvamento e exportação

1. **Adaptadores LoRA**: salvos localmente com `model.save_pretrained()` / `tokenizer.save_pretrained()` no Google Drive (`FINAL_MODEL_DIR`), preservando apenas os pesos treinados (poucos MB), não o modelo completo.
2. **Exportação para GGUF**: via `model.push_to_hub_gguf(...)`, a Unsloth mescla os adaptadores LoRA de volta aos pesos base (merge), converte o resultado para o formato GGUF e quantiza em `q4_k_m` (~4-5 GB) — formato compatível com inferência local via `llama.cpp`, Ollama, LM Studio.
3. **Publicação**: o modelo final é enviado ao Hugging Face Hub sob o repositório configurado em `HF_REPO_ID`, exigindo um token de acesso com escopo de escrita (`HF_TOKEN`, configurado como secret no Colab).

## 7. Validação do resultado

Após o treino, o notebook testa o modelo com dois exemplos do mesmo caso clínico (Seção 9): um formulado como múltipla escolha e outro como pergunta aberta — permitindo verificar, antes de exportar, se o modelo aprendeu a responder adequadamente em ambos os formatos, sempre citando a fonte e recomendando acompanhamento médico.

## 8. Limitações conhecidas

- **60 passos de treino** é um número reduzido, adequado para demonstrar o processo dentro do tempo de uma sessão gratuita do Colab — não necessariamente suficiente para maximizar a qualidade do modelo. Aumentar `MAX_STEPS` (ou treinar por múltiplas épocas) tende a melhorar a aderência ao formato e a qualidade das respostas, ao custo de mais tempo de GPU.
- O dataset (`Larxel/healthqa-br`) não possui um campo de explicação/raciocínio clínico — a "justificativa" usada no treino é apenas o texto da alternativa correta, não um raciocínio elaborado. O modelo aprende a citar a alternativa certa, mas não necessariamente a explicar o racional clínico em profundidade.
- Por ser derivado de um checkpoint **base** (não `-Instruct`), o comportamento de seguir instruções fora do padrão exato do prompt de treino não é garantido — por isso a camada de guardrails em código (Seção 10 do notebook) reforça, de forma determinística, a citação de fonte e a recomendação médica, complementando o que foi aprendido no treino.

## 9. Exemplo de dados

O dataset de treino (`Larxel/healthqa-br`) já é público e não contém dados de pacientes reais — são questões de provas médicas. Já os dados usados na demonstração do pipeline clínico (Seção 10 do notebook) são **sintéticos**, criados especificamente para este projeto. Abaixo, um exemplo de cada.

### 9.1 Exemplo bruto do dataset de treino

Registro original (`Larxel/healthqa-br`, split `train`, linha 0):

```json
{
  "id": "7f537911",
  "source": "Revalida",
  "year": 2013,
  "group": null,
  "question": "Homem com 49 anos de idade apresenta, há um ano e meio, quadro recorrente de monoartrite aguda, durando cada episódio cerca de três a cinco dias. Inicialmente foi acometido o joelho esquerdo, posteriormente o direito, em seguida o tornozelo direito e, há três semanas, houve recorrência do quadro no joelho esquerdo. Refere alívio dos sintomas com o uso de diclofenaco, que toma por conta própria. [...] O paciente é hipertenso e diabético há dez anos, em uso de hidroclorotiazida 25 mg/dia e glibenclamida 10 mg/dia. Refere tabagismo (5 cigarros/dia) e etilismo (cerveja, especialmente nos finais de semana).\n\nO diagnóstico do paciente e a conduta inicial a ser adotada são, respectivamente:\n\nA: gota não tofácea; realizar artrocentese e iniciar o uso de alopurinol imediatamente.\nB: artrite séptica; realizar artrocentese e aguardar a análise laboratorial do líquido sinovial.\nC: gota não tofácea; não realizar artrocentese e manter o uso de anti-inflamatório não hormonal.\nD: osteoartrite; solicitar radiografia dos joelhos e iniciar o uso de anti-inflamatório não hormonal.\nE: artrite séptica; não há necessidade de exames complementares e deve-se iniciar antibioticoterapia imediatamente.",
  "answer": "C"
}
```

### 9.2 O mesmo exemplo após a formatação (`format_example`, Seção 5)

**Como múltipla escolha** (mantém as alternativas, resposta começa com a letra):

```
### Questão:
Homem com 49 anos de idade apresenta [...] etilismo (cerveja, especialmente nos finais de semana).

O diagnóstico do paciente e a conduta inicial a ser adotada são, respectivamente:

A: gota não tofácea; realizar artrocentese e iniciar o uso de alopurinol imediatamente.
B: artrite séptica; realizar artrocentese e aguardar a análise laboratorial do líquido sinovial.
C: gota não tofácea; não realizar artrocentese e manter o uso de anti-inflamatório não hormonal.
D: osteoartrite; solicitar radiografia dos joelhos e iniciar o uso de anti-inflamatório não hormonal.
E: artrite séptica; não há necessidade de exames complementares e deve-se iniciar antibioticoterapia imediatamente.

### Resposta:
Alternativa correta: C — gota não tofácea; não realizar artrocentese e manter o uso de anti-inflamatório não hormonal.

Fonte: Revalida (2013).

Este conteúdo tem caráter educacional e não substitui a avaliação de um profissional de saúde. Consulte sempre um médico para confirmação diagnóstica e definição da conduta.
```

**Como pergunta aberta** (alternativas removidas pela `remover_alternativas()`, resposta em texto livre):

```
### Questão:
Homem com 49 anos de idade apresenta [...] etilismo (cerveja, especialmente nos finais de semana).

O diagnóstico do paciente e a conduta inicial a ser adotada são, respectivamente?

### Resposta:
gota não tofácea; não realizar artrocentese e manter o uso de anti-inflamatório não hormonal.

Fonte: Revalida (2013).

Este conteúdo tem caráter educacional e não substitui a avaliação de um profissional de saúde. Consulte sempre um médico para confirmação diagnóstica e definição da conduta.
```

### 9.3 Dados sintéticos (prontuários simulados, Seção 10)

Usados apenas para demonstrar o pipeline clínico (consulta ao prontuário + guardrails); não representam pacientes reais:

```python
prontuarios_df = pd.DataFrame([
    {
        "paciente_id": "P001",
        "idade": 62,
        "condicoes": "Diabetes tipo 2, Hipertensão",
        "exames_pendentes": ["Hemoglobina glicada (HbA1c)", "Perfil lipídico"],
        "exames_realizados": {"Glicemia de jejum": "182 mg/dL", "Creatinina": "1.1 mg/dL"},
        "medicacoes_atuais": ["Metformina 850mg 2x/dia"],
    },
    {
        "paciente_id": "P002",
        "idade": 45,
        "condicoes": "Asma",
        "exames_pendentes": [],
        "exames_realizados": {"Espirometria": "Normal"},
        "medicacoes_atuais": ["Salbutamol conforme necessidade"],
    },
])
```
