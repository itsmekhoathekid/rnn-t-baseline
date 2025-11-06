import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from .loss import RNNTLoss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        self.config = config

        # Build encoder & decoder
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

        # Build joint network
        self.joint = JointNet(
            input_size=config["joint"]["input_size"],
            inner_dim=config["joint"]["inner_size"],
            vocab_size=config["vocab_size"]
        )

        # Optionally share embedding weights
        if config.get("share_embedding", False):
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), \
                f"{self.decoder.embedding.weight.size(1)} != {self.joint.project_layer.weight.size(1)}"
            self.joint.project_layer.weight = self.decoder.embedding.weight

        # Loss function
        self.crit = RNNTLoss(
            blank=config.get("blank", 0),
            reduction=config.get("reduction", "mean")
        )

        self.blank = config.get("blank", 4)
        self.sos = config.get("sos", 1)
        self.eos = config.get("eos", 2)

    def forward(self, inputs, inputs_length, targets, targets_length):

        
        enc_state, _ = self.encoder(inputs, inputs_length)
        
        dec_state, _ = self.decoder(targets, targets_length.cpu())

        # Joint network
        logits = self.joint(enc_state, dec_state)

        # # Loss
        # loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())
        return logits

    def recognize(self, inputs, inputs_length):
        batch_size = inputs.size(0)
        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0).item()

                # print(pred)
                if pred == 2: 
                    break

                if pred not in (0, 1, 2, 4):
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list
        
        results = [decode(enc_states[i], inputs_length[i]) for i in range(batch_size)]

        return results

    @torch.no_grad()
    def greedy_batch_1(self, inputs, input_lengths, max_symbols_per_step=None):
        """
        Batched greedy inference for Transducer model.
        
        Args:
            enc_out: Tensor [B, T, D_enc] — encoder outputs
            enc_lens: Tensor [B] — valid time lengths
            max_symbols_per_step: int or None — optional limit for symbols per time step
        Returns:
            labels: List[List[int]] — predicted sequences (one per batch item)
        """
        enc_out, _ = self.encoder(inputs, input_lengths)
        enc_lens = input_lengths
        device = enc_out.device
        B, T, D_enc = enc_out.shape

        # init decoder
        hidden = None
        blank_id = getattr(self, "blank", 4)
        sos_id = getattr(self, "sos", blank_id)
        eos_id = getattr(self, "eos", -1)

        # buffers
        last_label = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        labels = [[] for _ in range(B)]
        blank_mask = torch.zeros(B, dtype=torch.bool, device=device)

        max_T = int(enc_lens.max())

        for t in range(max_T):
            f = enc_out[:, t:t+1, :]  # [B, 1, D]
            not_finished = ~blank_mask
            if not not_finished.any():
                break

            # reset for each time step
            symbols_added = 0
            still_running = True

            while still_running and (max_symbols_per_step is None or symbols_added < max_symbols_per_step):
                # decode step
                g, hidden_new = self.decoder(inputs = last_label, hidden = hidden)  # [B,1,D_dec]
                logit = self.joint(f, g)[:, 0, 0, :]               # [B,V]
                pred = logit.softmax(-1).argmax(-1)                # [B]

                blank_mask |= (pred == blank_id) | (t >= enc_lens)

                print(blank_mask, flush = True)
                # stop if all blanks
                if blank_mask.all():
                    still_running = False
                    break

                # update labels + hidden only for active samples
                for b in range(B):
                    if blank_mask[b]:
                        continue
                    token = pred[b].item()
                    print("token ", token, flush = True)
                    if token != blank_id and token != sos_id and token != eos_id:
                        labels[b].append(token)
                    last_label[b, 0] = token

                hidden = hidden_new
                symbols_added += 1
        
        return labels

    @torch.no_grad()
    def greedy_batch(self, inputs, input_lengths, max_output_len=200):
        # 1) Encode once for whole batch
        enc_out, _ = self.encoder(inputs, input_lengths)   # [B, T, D]

        B, T, D = enc_out.size()
        hidden = None

        # init decoder input: [B,1]
        tokens = torch.full((B,1), self.sos, dtype=torch.long, device=inputs.device)
        dec_state, hidden = self.decoder(tokens, hidden=hidden)        # [B,1,D_dec]

        # keep track finished
        finished = torch.zeros(B, dtype=torch.bool, device=inputs.device)
        results = [[] for _ in range(B)]

        t = 0
        while t < T and not finished.all():
            # 2) joint: enc_out[:,t,:] + last dec step 
            enc_step = enc_out[:, t, :].unsqueeze(1)       # [B,1,D]
            dec_step = dec_state[:, -1, :].unsqueeze(1)    # [B,1,D]
            logits = self.joint(enc_step, dec_step)        # [B,1,V]
            preds = logits.softmax(-1).argmax(dim=-1)      # [B,1]
            preds = preds.squeeze(1)                       # [B]

            # 3) for batch: update tokens one by one
            for b in range(B):
                if finished[b]:
                    continue

                p = preds[b].item()

                if p == self.eos:
                    finished[b] = True
                    continue

                if p not in [self.blank, self.sos]:
                    results[b].append(p)

                    # 1) token mới cho mẫu b (giữ batch=1)
                    token = torch.tensor([[p]], device=inputs.device)

                    # 2) Lấy hidden của mẫu b và ép contiguous
                    h, c = hidden
                    h_b = h.narrow(1, b, 1).contiguous()  # [num_layers, 1, H]
                    c_b = c.narrow(1, b, 1).contiguous()
                    hidden_b = (h_b, c_b)

                    # 3) Gọi decoder với keyword args để không nhầm 'length'
                    dec_state_b, hidden_b = self.decoder(inputs=token, hidden=hidden_b)

                    # 4) Ghi ngược lại vào hidden toàn batch bằng copy_
                    h[:, b:b+1, :].copy_(hidden_b[0])
                    c[:, b:b+1, :].copy_(hidden_b[1])
                    hidden = (h, c)

                    # 5) Cập nhật dec_state cho mẫu b (giữ đúng shape)
                    dec_state[b:b+1, -1:, :].copy_(dec_state_b[:, -1:, :])

            # time step only advances when blank or eos
            advance_mask = (preds == self.blank) | (preds == self.eos)
            if advance_mask.any():
                t += 1

        return results
    # @torch.no_grad()
    # def greedy_batch(self, inputs, inputs_length, max_len = 100):
    #     """
    #     Batched greedy inference for Transducer model.
        
    #     Args:
    #         enc_out: Tensor [B, T, D_enc] — encoder outputs
    #         enc_lens: Tensor [B] — valid time lengths
    #         max_symbols_per_step: int or None — optional limit for symbols per time step
    #     Returns:
    #         labels: List[List[int]] — predicted sequences (one per batch item)
    #     """
        
    #     enc_out, _ = self.encoder(inputs, inputs_length)
    #     enc_lens = inputs_length
    #     device = enc_out.device
    #     B, T, D_enc = enc_out.shape

    #     dec_input = torch.full((B, 1), self.sos, dtype=torch.long, device=device)
    #     for t in range(max_len):
    #         dec_out, _ = self.decoder(dec_input)
    #         logits = self.joint(enc_out, dec_out)[:, 0, 0, :]
    #         logp = F.softmax(logits, dim=-1)
    #         next_tokens = torch.argmax(logp, dim=-1)  # [B, T, U]
            
    #         k_is_blank = next_tokens == self.blank 
    #         raise


        

    # def recognize(self, inputs, inputs_length):
    #     batch_size = inputs.size(0)
    #     enc_states, _ = self.encoder(inputs, inputs_length)

            
    #     zero_token = torch.LongTensor([[0]])
    #     if inputs.is_cuda:
    #         zero_token = zero_token.cuda()

    #     def decode(enc_state, lengths):
    #         token_list = []
    #         dec_state, hidden = self.decoder(zero_token)

    #         for t in range(lengths):
    #             logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
    #             out = F.log_softmax(logits, dim=0).detach()
    #             pred = torch.argmax(out, dim=0).item()
                
    #             if pred == 2: 
    #                 break

    #             if pred not in (0,1,2,4) and pred != token_list[-1] if token_list else False:
    #                 token_list.append(pred)
    #                 token = torch.LongTensor([[pred]])
    #                 if enc_state.is_cuda:
    #                     token = token.cuda()
    #                 dec_state, hidden = self.decoder(token, hidden=hidden)

    #         return token_list
        
    #     results = [decode(enc_states[i], inputs_length[i]) for i in range(batch_size)]
    #     return results
