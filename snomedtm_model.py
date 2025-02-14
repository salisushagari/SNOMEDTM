import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from itertools import chain
import matplotlib.pyplot as plt
import os
import math

class SNOMEDTransformerConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=16,
        num_hidden_layers=16,
        intermediate_size=3072,
        vocab_size=30522,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

class SNOMEDTransformerModel(PreTrainedModel):
    config_class = SNOMEDTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
            ),
            num_layers=config.num_hidden_layers,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Embedding input tokens
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)

        # Passing through the encoder
        encoded = self.encoder(embeddings.permute(1, 0, 2))  # TransformerEncoder expects (seq_len, batch, hidden_size)
        encoded = encoded.permute(1, 0, 2)  # Back to (batch, seq_len, hidden_size)

        # Predict logits
        logits = self.lm_head(encoded)

        #loss = None
        if labels is not None:
            # Compute loss if labels are provided
            loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            

        return logits, loss
    
        
    def save_model(self, output_dir):
        """
        Save the model to the specified directory.
        Args:
            output_dir: Directory where the model will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


    def display_loss_curve(self, training_losses, title, save_path=None):
        """
        Display the training loss curve and save it as an image.
        Args:
            save_path: Path to save the loss curve image.

        """
        training_losses = [
        loss.item() if torch.is_tensor(loss) else loss for loss in training_losses
        ]
        title=title
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label=title, color='blue')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss curve saved to {save_path}")
        else:
            plt.show()
    
    def display_loss_curve_all(self, training_losses, evaluation_losses, training_accuracies, evaluation_accuracies, save_path=None):
        """
        Display the training loss, evaluation loss, training accuracy, and evaluation accuracy curves, and save it as an image.
    
        Args:
            training_losses (list): List of training losses.
            evaluation_losses (list): List of evaluation losses.
            training_accuracies (list): List of training accuracies.
            evaluation_accuracies (list): List of evaluation accuracies.
            save_path (str, optional): Path to save the loss curve image.
        """
        
        training_losses = [loss.item() if torch.is_tensor(loss) else loss for loss in training_losses]
        evaluation_losses = [loss.item() if torch.is_tensor(loss) else loss for loss in evaluation_losses]
        training_accuracies = [acc.item() if torch.is_tensor(acc) else acc for acc in training_accuracies]
        evaluation_accuracies = [acc.item() if torch.is_tensor(acc) else acc for acc in evaluation_accuracies]
    
        plt.figure(figsize=(10, 6))
    
        # Plot losses
        plt.plot(training_losses, label="Training Loss", color='blue')
        plt.plot(evaluation_losses, label="Evaluation Loss", color='red')
    
        # Plot accuracies
        plt.plot(training_accuracies, label="Training Accuracy", color='green')
        plt.plot(evaluation_accuracies, label="Evaluation Accuracy", color='orange')
    
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title("Pretraining Loss and Accuracy")
        plt.legend()
    
        if save_path:
            plt.savefig(save_path)
    
        plt.show()


