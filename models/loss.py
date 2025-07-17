from torchaudio.functional import rnnt_loss
import torch

class RNNTLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, logits, targets, fbank_len, text_len):
        # logits: [B, T, U, vocab_size]
        # targets: [B, U]
        # fbank_len: [B]
        # text_len: [B]


        loss = rnnt_loss(logits, targets[:, :].int(), fbank_len.int(), text_len.int(), blank=self.blank)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    

from warp_rna import rna_loss
        
class RNALoss(torch.nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(RNALoss, self).__init__()
        self.blank=blank
        self.reduction=reduction
    
    def forward(self, logits, targets, fbank_len, text_len): 
        """
        Args: 
            logits: [B, T, U, vocab_size]
            targets: [B, U]
            fbank_len: [B]
            text_len: [B]
        """

        # print("logits shape", logits.shape)
        # print("targets shape", targets.shape)
        # print("fbank_len shape", fbank_len.shape)
        # print("text_len shape", text_len.shape)
        # raise
        logits = logits.log_softmax(dim=-1)
        loss = rna_loss(logits, targets[:, :].int(), fbank_len.int(), text_len.int(), blank=self.blank)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss