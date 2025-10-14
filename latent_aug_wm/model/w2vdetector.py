import torch
import torch.nn as nn

from transformers import Wav2Vec2ForSequenceClassification
from peft import get_peft_model, LoraConfig


def load_peft_w2v(module: Wav2Vec2ForSequenceClassification):
    linear_layer_names = [
        n
        for n, m in module.named_modules()
        if ("attention.k_proj" in n or "attention.q_proj" in n) and ("encoder" in n)
    ]
    config = LoraConfig(
        target_modules=linear_layer_names,
        modules_to_save=[],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def create_w2v_detector(model_name="facebook/wav2vec2-base-960h", num_labels=2):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    peft_model = load_peft_w2v(model)
    return peft_model


if __name__ == "__main__":
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h", num_labels=2
    )
    peft_model = load_peft_w2v(model).cuda()

    for n, m in peft_model.named_modules():
        print(n, type(m))
    exit()
    import torchaudio

    transforms = torchaudio.transforms.Resample(24000, 16000).cuda()

    a = torch.randn((10, 48000)).cuda()
    a = transforms(a)
    print(a.shape)
    lebels = torch.ones((10), dtype=torch.long).cuda()

    out = peft_model(a, labels=lebels)
    print(out.logits)
    print(out.loss)
