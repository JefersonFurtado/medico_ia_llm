# Médico IA LLM

Fine-tuning de um modelo de linguagem (Llama-3-8B / Mistral-7B) para questões médicas em **português**, usando QLoRA (via [Unsloth](https://github.com/unslothai/unsloth)) sobre o dataset [Larxel/healthqa-br](https://huggingface.co/datasets/Larxel/healthqa-br), com um pipeline clínico construído com LangChain/LangGraph e guardrails de segurança reforçados em código.

## Conteúdo do repositório

- `medllm_healthqa_br_colab.ipynb` — notebook Google Colab com todo o fluxo do projeto:
  0. **Atalho (opcional)**: baixa e roda o modelo já treinado direto do Hugging Face, sem refazer o fine-tuning. Detecta GPU automaticamente e instala sempre wheels pré-compiladas (nunca compila do zero, evitando tempos de instalação de mais de 1 hora).
  1. **Setup do ambiente**: instalação do Unsloth, LangChain e LangGraph; montagem do Google Drive para checkpoints.
  2. **Carregamento do modelo**: Llama-3-8B (ou Mistral-7B) em 4-bit, com LoRA aplicado (otimizado para GPU T4).
  3. **Dataset**: [Larxel/healthqa-br](https://huggingface.co/datasets/Larxel/healthqa-br) — questões de múltipla escolha de provas de residência/revalidação médica no Brasil (ex.: Revalida). A resposta de treino é construída para sempre incluir a alternativa correta, a fonte da questão (`source` + `year`) e uma recomendação de acompanhamento médico.
  4. **Treinamento**: fine-tuning supervisionado (SFT) via TRL.
  5. **Exportação**: conversão do adapter para GGUF 4-bit (uso local via Ollama) e upload para o Hugging Face Hub.
  6. **Integração clínica**: pipeline em LangChain que contextualiza respostas com dados simulados de prontuário e orquestra o fluxo com LangGraph.

## Requisito de segurança e validação

O notebook atende aos três pontos abaixo em **duas camadas** — no comportamento aprendido pelo modelo durante o treino e, de forma determinística, em código:

| Requisito | Na fonte de treino (Seção 5) | Em código (Seção 10) |
|---|---|---|
| Limites de atuação (nunca prescrever/decidir sozinho) | Resposta de treino nunca é uma prescrição direta, sempre condicionada a validação | `garantir_limites_de_atuacao`: detecta linguagem de prescrição direta e anexa aviso de validação humana |
| Logging detalhado (auditoria) | — | `registrar_log`: grava timestamp + etapa + dados em cada passo do fluxo (consulta ao prontuário, guardrail aplicado, resposta gerada, cada nó do LangGraph) |
| Explainability (fonte da informação) | Toda resposta de treino termina em `Fonte: {source} ({year})` | `garantir_fonte`: se a resposta não citar "Fonte:", uma é adicionada automaticamente |
| Recomendação médica sempre presente | Toda resposta de treino termina com a frase fixa de recomendação médica | `garantir_recomendacao_medica`: se ausente, a frase padrão é adicionada |

## Modelo treinado

Após rodar o pipeline completo (Seções 1 a 8), o modelo resultante é publicado no Hugging Face Hub sob o `HF_REPO_ID` configurado no notebook (padrão: `jeferson2106/medico_ia_jeferson_br` — ajuste para o seu usuário).

- Base: Llama 3 (8B)
- Formato: GGUF (quantizado, ex. Q4_K_M), gerado com Unsloth
- Uso local: compatível com llama.cpp, Ollama, LM Studio e Docker

## Aviso

Este projeto tem finalidade educacional/experimental. As respostas geradas não substituem avaliação médica profissional — o pipeline sempre inclui a recomendação de acompanhamento médico e guardrails para impedir prescrições diretas sem validação de um médico.

## Como usar

1. Abra `medllm_healthqa_br_colab.ipynb` no Google Colab (GPU T4 recomendada).
2. Configure a secret `HF_TOKEN` no Colab (ícone de chave 🔑) com um token do Hugging Face com escopo **Write**, necessário para publicar o modelo treinado.
3. Rode as células em ordem, Seções 1 a 8, para fazer o fine-tuning e publicar seu próprio modelo.
4. Depois de publicado, use a **Seção 0** para baixar e testar o modelo já treinado a qualquer momento, sem repetir o fine-tuning (funciona com ou sem GPU).
