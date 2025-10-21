# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as f
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip import clip

import torch

class AlphaModule(nn.Module):
    def __init__(self, shape):
        super(AlphaModule, self).__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.alpha = Parameter(torch.rand(tuple([1] + list(shape))) * 0.1,
                               requires_grad=True)

    def forward(self, x):
        return x * self.alpha

    def parameters(self, recurse: bool = True):
        yield self.alpha


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.idx = 0
        for module in args:
            self.add_module(str(self.idx), module)
            self.idx += 1

    def append(self, module):
        self.add_module(str(self.idx), module)
        self.idx += 1

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.idx
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    


def detokenize(tokens: torch.Tensor):
    """
    Converts a tensor of token ids back into a list of strings.

    Parameters
    ----------
    tokens : torch.Tensor
        A two-dimensional tensor containing token ids.

    Returns
    -------
    List[str]
        A list of strings where each string is the detokenized representation of the corresponding row in the input tensor.
    """
    _tokenizer = _Tokenizer()
    detokenized_texts = []
    for token_list in tokens:
        text_tokens = []
        for token_id in token_list:
            token_id = token_id.item()  # Convert PyTorch tensor to Python integer
            if token_id == _tokenizer.encode("") or token_id == 0:
                # Skip the special start/end tokens and padding tokens (assuming 0 is used for padding)
                continue
            token = _tokenizer.decode(token_id)  # Get the string representation of the token
            text_tokens.append(token.replace('</w>', ' '))  # Remove BPE end-of-word marker
        detokenized_text = ''.join(text_tokens).strip()
        detokenized_texts.append(detokenized_text)

    return detokenized_texts


def upsample(x):
    if isinstance(x, list):
        inputs0, inputs1 = x
        if inputs0.shape == inputs1.shape:
            x= torch.cat([inputs0, inputs1], dim=-1)
            return x,inputs0,inputs1
        if inputs1.dim() == 2:
            inputs0 = inputs0.unsqueeze(0)
            inputs1 = inputs1.unsqueeze(0)
        # upsample resolution and concat
        upsampled = torch.nn.Upsample(scale_factor=64, mode='nearest')(inputs1.transpose(2, 1)).transpose(2, 1)
        new_inputs0 = inputs0.reshape(upsampled.shape)
        x = torch.concatenate([new_inputs0, upsampled], dim=2)
        return x,new_inputs0, upsampled
    else:
        new_inputs0, upsampled = torch.chunk(x, 2, dim=x.dim()-1)
        return x,new_inputs0,upsampled

class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, feats):
        device = feats.device# N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x bag_prediction
        device = feats.device
        value = self.v(feats)  # N x value, unsorted
        query = self.q(feats).view(feats.shape[0], -1)  # N x query, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in
        # shape N x bag_prediction
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # critical instances, m_feats in shape bag_prediction x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape bag_prediction x query
        attention = torch.mm(query, q_max.transpose(0, 1))  # compute inner product of query to each
        # entry of q_max, attention in shape N x bag_prediction, each column contains unnormalized attention scores
        attention = attention / torch.sqrt(torch.tensor(query.shape[1], dtype=torch.float32, device=device))
        attention = f.softmax(attention, 0)  # normalize attention scores, attention in shape N x bag_prediction,
        bag = torch.mm(attention.transpose(0, 1), value)  # compute bag representation, bag in shape bag_prediction x
        # value

        bag = bag.view(1, bag.shape[0], bag.shape[1])  # 1 x bag_prediction x value
        bag_prediction = self.fcc(bag)  # 1 x bag_prediction x 1
        bag_prediction = bag_prediction.view(1, -1)
        return bag_prediction, attention, bag


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier, args):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.args = args

    def forward(self, x):
        if self.args.dsmil_multiscale:
            if isinstance(x, list):
                inputs0, inputs1 = x
                sampled = torch.nn.Upsample(scale_factor=64, mode='nearest')(inputs1.transpose(2, 1)).transpose(2, 1)
                new_inputs0 = inputs0.reshape(sampled.shape)
                x = torch.concatenate([new_inputs0, sampled], dim=2)
                #x= new_inputs0+sampled
            else:
                new_inputs0,sampled=x,x
        else:
            if isinstance(x, list):
                inputs0, inputs1 = x
                new_inputs0, sampled = inputs0,inputs0
                x = inputs0.view(-1, self.args.input_size)
            else:
                new_inputs0, sampled = x,x
        x = x.squeeze(0)
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        y_prob = f.softmax(prediction_bag, dim=1)
        y_hat = torch.argmax(prediction_bag)
        proj= new_inputs0+sampled
        return prediction_bag, y_prob, y_hat, A, classes, B,proj


class FCLayerOptimized(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayerOptimized, self).__init__()
        # self.fc = nn.Sequential(nn.Linear(in_size, out_size))
        self.fc = nn.Sequential(nn.Linear(in_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh(),
                                nn.Linear(128, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifierOptimized(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifierOptimized, self).__init__()
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, feats):
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifierOptimized(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifierOptimized, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C,
        # each column contains unnormalized attention scores
        A = f.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNetOptimized(nn.Module):
    def __init__(self, i_classifier, b_classifier,args):
        super(MILNetOptimized, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.args=args

    def forward(self, x):
        if self.args.dsmil_multiscale:
            x,_,_=upsample(x)
        x=x.squeeze(0)
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        Y_prob = f.softmax(prediction_bag, dim=1)
        Y_hat = torch.argmax(prediction_bag)
        return prediction_bag, Y_prob, Y_hat, A, classes, B



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # conch
        #self.transformer = clip_model.text
        self.transformer = clip_model.text.transformer
        self.positional_embedding = clip_model.text.positional_embedding
        self.ln_final = clip_model.text.ln_final
        self.text_projection = clip_model.text.text_projection

        # plip
        #self.transformer = clip_model.text_model.encoder
        #self.positional_embedding = clip_model.text_model.embeddings.position_embedding.weight
        #self.ln_final = clip_model.text_model.final_layer_norm
        #self.text_projection = torch.Tensor(clip_model.text_projection.weight).cuda()
        #self.task_keys=torch
        #self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # conch
        x = prompts + self.positional_embedding.to(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(torch.float32)    
        x = x[torch.arange(x.shape[0]), torch.where(tokenized_prompts == 2)[1]] @ self.text_projection
        return x
    
        # plip
        #x = prompts + self.positional_embedding.type(self.dtype)
        #x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer(x).last_hidden_state
        #x = x.permute(1, 0, 2)  # LND -> NLD
        #x = self.ln_final(x).type(self.dtype)

        #x = self.transformer(prompts).last_hidden_state
        #x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.T

        #return x


class MLCPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model,args):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls
        n_ctx_pos = args.n_ctx
        n_ctx_neg = args.n_ctx
        self.old_pos_prompts = []
        self.old_neg_prompts = []
        self.tokenizer = _Tokenizer()
        dtype = clip_model.dtype
        ctx_dim = clip_model.visual_projection.weight.shape[0]
        print("Initializing class-specific contexts")
        ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        #prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        #prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        prompt_prefix_pos= "Image of "
        prompt_prefix_neg = "Image of "

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.text_model.embeddings.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.text_model.embeddings.token_embedding(tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def freeze_prompts(self, task, tot_classes):
        import copy
        for i in range(task * tot_classes, task * tot_classes + tot_classes):
            self.old_pos_prompts.append(copy.deepcopy(self.ctx_pos[i].detach()))
            self.old_neg_prompts.append(copy.deepcopy(self.ctx_neg[i].detach()))
            print("Freezing prompts", str(i))

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        if len(self.old_pos_prompts) > 0:
            ctx_pos = ctx_pos[len(self.old_pos_prompts):]
            ctx_neg = ctx_neg[len(self.old_neg_prompts):]
            ctx_pos = torch.cat([torch.stack(self.old_pos_prompts), ctx_pos])
            ctx_neg = torch.cat([torch.stack(self.old_neg_prompts), ctx_neg])

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        #detokenize(prefix_neg)
        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)
        tokenized_prompts = self.tokenized_prompts
        return prompts, tokenized_prompts

