class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

    def forward(self):
        pass

class Transformer(pytorch.)
    def predict(self) -> :
        for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            
            pred = model(input_sequence, y_input, tgt_mask)
            
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()