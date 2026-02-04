import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from peft import get_peft_model, LoraConfig


def load_peft_w2v(module: Wav2Vec2ForSequenceClassification):
    linear_layer_names = [
        n
        for n, m in module.named_modules()
        if ("attention.k_proj" in n or "attention.q_proj" in n) and ("encoder" in n)
    ]
    config = LoraConfig(
        target_modules=linear_layer_names,
        # modules_to_save=['classifier.bias', 'classifier.weight', 'projector.bias', 'projector.weight', 'wav2vec2.masked_spec_embed'],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def load_peft_w2v_attent(module):

    config = LoraConfig(
        target_modules=[
            "k_proj",
            "v_proj",
            "q_proj",
            "intermediate_dense",
            "output_dense",
        ],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def create_w2v_detector(
    model_name="facebook/wav2vec2-base-960h",
    num_labels=2,
    attn_implementation="flash_attention_2",
):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, attn_implementation=attn_implementation
    )
    peft_model = load_peft_w2v(model)
    return peft_model


def create_full_trainable_param_w2v(
    model_name="facebook/wav2vec2-base-960h",
    num_labels=2,
    attn_implementation="flash_attention_2",
):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, attn_implementation=attn_implementation
    )
    return model


def create_w2v_encoder_peft(
    model_name="facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2"
):
    if "flash_attention" in attn_implementation:
        model = Wav2Vec2Model.from_pretrained(
            model_name, attn_implementation=attn_implementation, dtype=torch.bfloat16
        )
    else:
        model = Wav2Vec2Model.from_pretrained(
            model_name, attn_implementation=attn_implementation
        )
    peft_model = load_peft_w2v_attent(model)
    return peft_model


def create_w2v_encoder(
    model_name="facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2"
):
    if "flash_attention" in attn_implementation:
        model = Wav2Vec2Model.from_pretrained(
            model_name, attn_implementation=attn_implementation, dtype=torch.bfloat16
        )
    else:
        model = Wav2Vec2Model.from_pretrained(
            model_name, attn_implementation=attn_implementation
        )
    return model


class SimpleLinearClassifier(nn.Module):
    def __init__(self, latent_dim=768, num_label=2):
        super().__init__()
        self.layer = nn.Linear(latent_dim, num_label)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, hidden_states, labels):
        out = self.forward(hidden_states)
        loss = self.criterion(out, labels)
        softmax_label = F.softmax(out, dim=-1)
        return softmax_label, loss

    def forward(self, hidden_states):
        latent = hidden_states.mean(1)  # B, seq_len, N
        out = self.layer(latent)
        return out


if __name__ == "__main__":
    # peft_model = create_w2v_detector().cuda()
    attn_implementation = "flash_attention_2"
    model_name = "facebook/wav2vec2-xls-r-300m"
    model = Wav2Vec2Model.from_pretrained(
        model_name, attn_implementation=attn_implementation, dtype=torch.bfloat16
    )
    peft_model = load_peft_w2v_attent(model).cuda()
    # for k, v in peft_model.named_parameters():
    #     print(k, type(v))
    exit()
    # peft_model = load_peft_w2v(model).cuda()
    classifier = SimpleLinearClassifier().cuda()
    # for k, v in peft_model.named_parameters():
    #     print(k, type(v))
    # exit()
    from torch.autograd import Variable
    import torchaudio

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        transforms = torchaudio.transforms.Resample(24000, 16000).cuda()

        a = Variable(torch.randn((10, 48000))).cuda()
        a.requires_grad = True
        a = transforms(a)
        print(a.shape)
        lebels = torch.ones((10), dtype=torch.long).cuda()

        # out = peft_model(a, labels=lebels)
        out = peft_model(a)
        print(out.last_hidden_state.shape)  # B, seq_len, latent_dim
        print(out.last_hidden_state.mean(1).shape)  # # B, seq_len, latent_dim
        print(classifier.compute_loss(out.last_hidden_state, lebels))
