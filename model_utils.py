from transformers import AutoConfig
import torch.nn as nn
from layers import convert_to_sn, remove_all_normalization_layers

def freeze_and_modify_weights(model, config):
    if config.freeze_embeddings == 'True':
        print(f'FREEZING EMBEDDINGS')
        if 'deberta' in config.base_model:
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
        elif 'roberta' in config.base_model:
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
        elif 'longformer' in config.base_model:
            for param in model.longformer.embeddings.parameters():
                param.requires_grad = False

    if config.freeze_layers > 0:
        print(f'FREEZING {config.freeze_layers} LAYERS')
        if 'deberta' in config.base_model:
            for layer in model.deberta.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif 'roberta' in config.base_model:
            for layer in model.roberta.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif 'longformer' in config.base_model:
            for layer in model.longformer.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    if config.reinit_layers > 0:
        print(f'REINITIALIZING LAST {config.reinit_layers} LAYERS')
        model_config = AutoConfig.from_pretrained(config.base_model)
        if 'deberta' in config.base_model:
            for layer in model.deberta.encoder.layer[-config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
        if 'roberta' in config.base_model:
            for layer in model.roberta.encoder.layer[-config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
        if 'longformer' in config.base_model:
            for layer in model.longformer.encoder.layer[-config.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
    
    if config.sigma_reparam == 'True':
        print("CONVERTING MODEL TO SigamReparam VARIENT")
        model = remove_all_normalization_layers(convert_to_sn(model))
    
    return model

