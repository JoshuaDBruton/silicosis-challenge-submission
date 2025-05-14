import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super().__init__()
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, embedding_dim)
        )

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_embeds, input_embeds], dim=1)


class PTunedTextEncoder(nn.Module):
    def __init__(self, base_text_encoder, tokenizer, prompt_length=10, context_length=77): # context_length kept for signature, but open_clip tokenizer may ignore it
        super().__init__()
        self.base_text_encoder = base_text_encoder
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

        self.embedding_layer = base_text_encoder.transformer.embeddings.word_embeddings
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.soft_prompt = SoftPrompt(self.prompt_length, self.embedding_dim)

        # Freeze the base text encoder
        for param in self.base_text_encoder.parameters():
            param.requires_grad = False

    def forward(self, texts):
        current_device = self.soft_prompt.soft_prompt.device

        input_ids = self.tokenizer(texts).to(current_device)

        pad_token_id = 0
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        original_attention_mask = (input_ids != pad_token_id).long()

        input_embeds_original = self.embedding_layer(input_ids)

        combined_input_embeds = self.soft_prompt(input_embeds_original)

        batch_size = combined_input_embeds.size(0)
        prompt_attention_mask = torch.ones(
            batch_size, self.prompt_length, dtype=torch.long, device=current_device
        )

        combined_attention_mask = torch.cat(
            [prompt_attention_mask, original_attention_mask], dim=1
        )

        outputs = self.base_text_encoder.transformer(
            inputs_embeds=combined_input_embeds,
            attention_mask=combined_attention_mask # ADDED THIS ARGUMENT
        )

        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs

        cls_token_index = self.prompt_length
        cls_embeds = last_hidden_state[:, cls_token_index, :]

        final_embeds = self.base_text_encoder.proj(cls_embeds)

        return final_embeds
