# Médico IA LLM

Fine-tuning de um modelo de linguagem (Llama-3-8B / Mistral-7B) para perguntas e respostas biomédicas, usando QLoRA (via [Unsloth](https://github.com/unslothai/unsloth)) e um pipeline clínico construído com LangChain/LangGraph.

## Conteúdo do repositório

- `medllm_pubmedqa_colab.ipynb` — notebook Google Colab com todo o fluxo do projeto:
  0. **Atalho (opcional)**: baixa e roda o modelo já treinado direto do Hugging Face, sem refazer o fine-tuning. Detecta automaticamente se há GPU disponível e ajusta a inferência (GPU ou CPU).
  1. **Setup do ambiente**: instalação do Unsloth, LangChain e LangGraph; montagem do Google Drive para checkpoints.
  2. **Carregamento do modelo**: Llama-3-8B (ou Mistral-7B) em 4-bit, com LoRA aplicado a ~0,52% dos parâmetros (otimizado para GPU T4).
  3. **Dataset**: [PubMedQA](https://pubmedqa.github.io/), formatado no padrão de prompt Alpaca (contexto, pergunta, resposta).
  4. **Treinamento**: fine-tuning supervisionado (SFT) via TRL, 60 steps (~16-25 min em T4).
  5. **Exportação**: conversão do adapter para GGUF 4-bit (uso local via Ollama) e upload para o Hugging Face Hub.
  6. **Integração clínica**: pipeline em LangChain que contextualiza respostas com dados simulados de prontuário, aplica guardrails (sem prescrição direta sem validação médica), registra logs de auditoria e orquestra o fluxo com LangGraph.

## Modelo treinado

O modelo resultante deste fine-tuning está publicado no Hugging Face Hub:

Link: https://huggingface.co/jeferson2106/medico_ia_jeferson

- Base: Llama 3 (8B)
- Formato: GGUF (quantizado, ex. Q4_K_M ~4,92 GB), gerado com Unsloth
- Uso local: compatível com llama.cpp, Ollama, LM Studio e Docker

## Configuração de treinamento

| Parâmetro | Valor |
|---|---|
| Batch size | 2 (efetivo 8 com gradient accumulation) |
| Learning rate | 2e-4 |
| Max steps | 60 |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0.0 |

## Aviso

Este projeto tem finalidade educacional/experimental. As respostas geradas não substituem avaliação médica profissional e o pipeline inclui guardrails para impedir prescrições diretas sem validação de um médico.

## Como usar

1. Abra `medllm_pubmedqa_colab.ipynb` no Google Colab.
2. Execute as células em ordem (GPU T4 recomendada).
3. Para usar o modelo já treinado sem refazer o fine-tuning, rode apenas a célula da **Seção 0** do notebook — ela baixa o modelo direto do Hugging Face ([jeferson2106/medico_ia_jeferson](https://huggingface.co/jeferson2106/medico_ia_jeferson)) e funciona com ou sem GPU.
