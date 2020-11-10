import config


def tokens_to_name(x, tokenizer):
    return "".join([tokenizer.itos[i] for i in x])


def formula_to_list(formula):
    if formula == "O":
        return ["O"]

    if formula == "c(NO2)3":
        formula = "B(NO2)3"

    form = []
    ele = []
    for char in formula:
        if char == ",":
            continue
        if char in "1234567890[]()-.·":
            if ele:
                form.append("".join(ele))
                ele = []
            form.append(char)
            continue
        elif char.islower():
            ele.append(char)
            form.append("".join(ele))
            ele = []
            continue
        elif char.isupper() and char not in ele and len(ele) > 0:
            form.append("".join(ele))
            ele = []
        elif char.isupper() and char in ele:
            continue
        ele.append(char)

    if ele and "".join(ele) != form[-1]:
        form.append("".join(ele))
    if "" in form:
        form.remove("")
    return form


class Batch:
    """
    Holds the src and targets along with their masks and targets
    """

    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.trg_y != pad_index
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if config.USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.formula, batch.name, pad_idx)

class LossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

def predict_output(model, src, src_mask, src_lengths, max_len=30, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
