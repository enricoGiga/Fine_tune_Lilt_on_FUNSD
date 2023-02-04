from transformers import AutoModel
import torch.nn as nn


class SERClassification(nn.Module):

    def __init__(self, num_classes):
        super().__init__()


        self.model = AutoModel.from_pretrained("C:\\Users\\enrico\\PycharmProjects\\fine_tune_Lilt_on_FUNSD\\data\\models\\Lilt", local_files_only=True)
        hidden_dim = self.model.config.hidden_size  ## 768
        self.cls_layer = nn.Sequential(nn.Linear(in_features=hidden_dim,
                                                 out_features=hidden_dim),
                                       nn.ReLU(),

                                       nn.Linear(in_features=hidden_dim, out_features=num_classes))

    def forward(self, batch):
        output = self.model(input_ids=batch["input_ids"],
                            bbox=batch["bbox"],
                            attention_mask=batch["attention_mask"],
                            ).last_hidden_state[:, :512, :]  ## The output is [none, 512, 768],

        output = self.cls_layer(output)

        return {"logits": output}