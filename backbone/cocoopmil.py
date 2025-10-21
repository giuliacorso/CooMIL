import copy
from backbone.dsmil import MILNetOriginal
import torch
import torch.nn as nn
import torch.nn.functional as f
import wandb
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import CLIPModel

from backbone.utils.modules import TextEncoder, upsample
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

__all__ = ['cocoopmil', 'CoCoopMil']



class MLCPromptLearner(nn.Module):
    def __init__(self, classnames, task_names, clip_model, args):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls
        n_ctx_tumor = args.n_ctx
        n_ctx_normal_tissue = args.n_ctx
        self.old_tumor_prompts = []
        self.old_normal_prompts = []
        self.tokenizer = get_tokenizer()
        self.args=args
        dtype = torch.float32
        ctx_dim = clip_model.text.ln_final.weight.shape[0]
        print("Initializing class-specific contexts")
        prompt_prefix_tumor = "An H&E image of a tumor"
        prompt_prefix_normal_tissue = "An H&E image of a normal"
        len_prefix_tumor = len(prompt_prefix_tumor.split(" "))
        if 'H&E' in prompt_prefix_tumor:
            len_prefix_tumor += 2
        len_prefix_normal = len(prompt_prefix_normal_tissue.split(" ")) 
        if 'H&E' in prompt_prefix_normal_tissue:
            len_prefix_normal += 2

        ctx_vectors_pos = torch.empty(n_cls, n_ctx_tumor, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(len(task_names), n_ctx_normal_tissue, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_tumor = nn.Parameter(ctx_vectors_pos)
        self.ctx_normal_tissue = nn.Parameter(ctx_vectors_neg)

        prompt_t = " ".join(["X"] * n_ctx_tumor)
        prompt_n = " ".join(["X"] * n_ctx_normal_tissue)

        print(f'Initial positive context: "{prompt_prefix_tumor}"')
        print(f'Initial negative  context: "{prompt_prefix_normal_tissue}"')
        print(f"Number of positive context words (tokens): {n_ctx_tumor}")
        print(f"Number of positive context words (tokens): {n_ctx_normal_tissue}")
        classnames = [name.replace("_", " ") for name in classnames]
        prompts_tumor = [prompt_prefix_tumor + " " + prompt_t + " " + name + "." for name in classnames]
        prompts_normal = [prompt_prefix_normal_tissue + " " + prompt_n + " " + name + " tissue." for name in task_names]
        tokenized_prompts_tumor = []
        tokenized_prompts_normal = []
        for prompt_tumor in prompts_tumor:
            tokenized_prompts_tumor.append(tokenize(texts=[prompt_tumor], tokenizer=self.tokenizer))
        for prompt_normal in prompts_normal:
            tokenized_prompts_normal.append(tokenize(texts=[prompt_normal], tokenizer=self.tokenizer))
        tokenized_prompts_tumor = torch.cat(tokenized_prompts_tumor)
        tokenized_prompts_normal = torch.cat(tokenized_prompts_normal)
        with torch.no_grad():
            embedding_tumor = clip_model.text.token_embedding(tokenized_prompts_tumor).type(dtype)
            embedding_normal = clip_model.text.token_embedding(tokenized_prompts_normal).type(dtype)

        self.register_buffer("token_prefix_tumor", embedding_tumor[:, :1+len_prefix_tumor, :])
        self.register_buffer("token_prefix_normal", embedding_normal[:, :1+len_prefix_normal, :])
        self.register_buffer("token_suffix_tumor", embedding_tumor[:, 1+len_prefix_tumor+n_ctx_tumor:, :])
        self.register_buffer("token_suffix_normal", embedding_normal[:, 1+len_prefix_normal+n_ctx_normal_tissue:, :])
        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_tumor
        tokenized_prompts = torch.cat([tokenized_prompts_normal, tokenized_prompts_tumor], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, task_metatoken, task_mask, add_context=False):
        ctx_tumor = self.ctx_tumor
        ctx_normal = self.ctx_normal_tissue

        if len(self.old_tumor_prompts) > 0:
            new_ctx_tumor = ctx_tumor[len(self.old_tumor_prompts):]
            new_ctx_normal = ctx_normal[len(self.old_normal_prompts):]
            ctx_tumor = torch.cat([torch.stack(self.old_tumor_prompts), new_ctx_tumor])
            ctx_normal = torch.cat([torch.stack(self.old_normal_prompts), new_ctx_normal])

        if self.args.add_context_only_learnable and add_context:
            if self.args.add_normal_context:
                ctx_normal=self.generate_context_prompts(task_metatoken, ctx_normal, task_mask)
            if self.args.add_tumor_context:
                ctx_tumor = self.generate_context_prompts(task_metatoken, ctx_tumor, task_mask)

        prefix_tumor = self.token_prefix_tumor
        suffix_tumor = self.token_suffix_tumor

        prefix_normal = self.token_prefix_normal
        suffix_normal = self.token_suffix_normal
       
        prompts_tumor = torch.cat(
            [
                prefix_tumor,  # (n_cls, 1, dim)
                ctx_tumor,  # (n_cls, n_ctx, dim)
                suffix_tumor,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_normal = torch.cat(
            [
                prefix_normal,
                ctx_normal,
                suffix_normal,
            ],
            dim=1,
        )

        if not self.args.add_context_only_learnable and add_context:
            if self.args.add_normal_context:
                prompts_normal = self.generate_context_prompts(task_metatoken, prompts_normal, task_mask)
            if self.args.add_tumor_context:
                prompts_tumor = self.generate_context_prompts(task_metatoken, prompts_tumor, task_mask)

        prompts = torch.cat([prompts_normal, prompts_tumor], dim=0)
        tokenized_prompts = self.tokenized_prompts
        return prompts, tokenized_prompts

    def generate_context_prompts(self, meta_tokens, prompts, task_mask):
        n_prompt = prompts.shape[0]
        n_meta = meta_tokens.shape[0]
        meta_tokens = meta_tokens.repeat_interleave(n_prompt//n_meta, dim=0).unsqueeze(1)
        if task_mask is not None:
            meta_tokens[task_mask, ...] = 0
        prompts = prompts + meta_tokens
        return prompts
    
    def freeze_prompts(self, task, tot_classes):
        import copy
        for i in range(task * tot_classes, task * tot_classes + tot_classes):
            freezed_tumor = copy.deepcopy(self.ctx_tumor[i].detach())
            freezed_tumor.requires_grad = False
            self.old_tumor_prompts.append(freezed_tumor)
            print("Freezing prompts", str(i))
        freezed_normal_tissue = copy.deepcopy(self.ctx_normal_tissue[task].detach())
        freezed_normal_tissue.requires_grad = False
        self.old_normal_prompts.append(freezed_normal_tissue)


class MetaNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MetaNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(nn.Linear(input_size, input_size),
                                 nn.ReLU(),
                                 nn.Linear(input_size, output_size),
                                 nn.Tanh())

    def forward(self, x):
        return self.net(x)


class CoCoopMil(nn.Module):
    def __init__(self, classnames, task_names, args):
        super().__init__()
        self.args = args
        clip_model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/work/H2020DeciderFicarra/gbontempo/conch_model.bin")
        self.prompt_learner = MLCPromptLearner(classnames, task_names, clip_model, args)
        self.visual_projection = torch.Tensor(clip_model.visual.proj_contrast).cuda().detach()
        self.visual_projection.requires_grad = False
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.text_encoder.requires_grad_(False)
        self.old_keys, self.old_metas, self.old_dsmil = [], [], []
        self.initial_prompts = None
        self.add_context = args.context
        self.dsmil_freezed = None
        self.logit_scale = nn.Parameter(torch.tensor(5.0))
        self.temperature=self.args.temperature
        input_size = args.input_size
        self.keys = nn.Parameter(nn.init.normal_(torch.empty(4, 512), std=0.02))
        self.clusters = nn.init.normal_(torch.empty(4 * args.n_clusters, input_size), std=0.02)
        self.dtype = torch.float32

        self.dsmil = MILNetOriginal(args, clip_model)
        if self.args.dsmil_multiscale:
            scale = 2
        else:
            scale = 1
        self.classes_per_task = self.args.n_classes_per_task
        self.n_tasks= self.args.n_classes//self.args.n_classes_per_task

        self.metanets= torch.nn.ModuleList([MetaNet(input_size * scale * self.classes_per_task, 768) for i in range(self.n_tasks)])

    def forward_dsmil_metanet(self, x):
        tot_past = len(self.old_dsmil)

        if tot_past == 0:
            if self.args.dsmil_freezed and self.dsmil_freezed is not None:
                prediction_bag, y_prob, y_hat, a, classes, bag, _, _ = self.dsmil_freezed(x)
                bag=bag.detach()
            else:
                prediction_bag, y_prob, y_hat, a, classes, bag, _, _ = self.dsmil(x)

            task_meta_tokens = self.create_task_meta_token(bag)
            return prediction_bag, y_prob, y_hat, a, classes, bag, task_meta_tokens
        else:
            # get old predictions, and meta tokens
            outputs = []
            old_bag_predictions = []
            for model, meta in zip(self.old_dsmil, self.old_metas):
                old_prediction_bag, _, _, _, classes, bag, _, _ = model(x)
                old_bag_predictions.append(old_prediction_bag)
                meta_tokens = self.create_task_meta_token(bag)
                outputs.append(meta_tokens)
            old_metatokens = torch.concatenate([item[task :(task + 1), :] for task, item in enumerate(outputs)], dim=0)
            # get new predictions and meta tokens
            new_prediction_bag, y_prob, y_hat, a, classes, bag, _, _ = self.dsmil(x)
            prediction_bag = self.combine_old_new_predictions(old_bag_predictions, new_prediction_bag)
            new_meta_tokens = self.create_task_meta_token(bag)
            new_meta_tokens = new_meta_tokens[ tot_past:, :]
            try:
                meta_tokens = torch.cat([old_metatokens, new_meta_tokens], dim=0)
            except:
                print("old", old_metatokens.shape)
            return prediction_bag, y_prob, y_hat, a, classes, bag, meta_tokens

    def create_class_meta_token(self, bag,metanet):
        meta_token = metanet(bag.squeeze(0))
        meta_tokens = meta_token / meta_token.norm(dim=-1, keepdim=True)
        meta_token = meta_tokens.unsqueeze(0)
        return meta_token

    def create_task_meta_token(self, bag):
        #troppe squeeze
        task_bag = bag
        task_bag = torch.chunk(task_bag, task_bag.shape[1]//self.classes_per_task, dim=1)

        task_metatokens = []

        for idx, tb in enumerate(task_bag):
            tb = tb.reshape(-1, tb.shape[1]*tb.shape[2])
            meta_token = self.metanets[idx](tb)
            meta_token = meta_token / meta_token.norm(dim=-1, keepdim=True)
            task_metatokens.append(meta_token)
        meta_tokens=torch.cat(task_metatokens,dim=0)
        return meta_tokens

    def freeze_keys(self, task):
        #self.old_keys.append(self.keys[task].detach().clone())
        print("Freezing keys", str(task))
        print("Freezing dsmil", str(task))
        print("Freezing metanet", str(task))
        old_dsmil = copy.deepcopy(self.dsmil)
        old_meta = copy.deepcopy(self.metanets[task])
        for param in old_dsmil.parameters():
            param.requires_grad = False
        for param in old_meta.parameters():
            param.requires_grad = False
        self.old_dsmil.append(old_dsmil)
        self.old_metas.append(old_meta)
        self.logit_scale= copy.deepcopy(self.logit_scale)
        self.logit_scale.requires_grad = False

    def combine_old_new_predictions(self, old_bag_predictions, new_bag_predictions):
        old_bag_predictions = torch.cat(
            [item[:, idx * 2:(idx + 1) * 2] for idx, item in enumerate(old_bag_predictions)], dim=1)
        new_bag_predictions = new_bag_predictions[:, len(self.old_dsmil) * 2:]
        bag_predictions = torch.cat([old_bag_predictions, new_bag_predictions], dim=1)
        return bag_predictions

    def aggregate(self, prompts, tokenized_prompts, inputs0, visual_projection=True):
        # get text features and normalize
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if visual_projection:
            #inputs0 = inputs0.view(-1, 768) @ self.visual_projection.T
            inputs0 = inputs0.view(-1, 512) @ self.visual_projection
        inputs0 = inputs0 / inputs0.norm(dim=-1, keepdim=True)

        # Class-Specific Region Feature Aggregation

        wandb.log({"logit_scale": self.logit_scale.exp().item()})
        output =self.logit_scale.exp()* f.conv1d(inputs0.unsqueeze(0).transpose(2, 1), text_features[:, :, None])
        normal_dist = output[0, :self.n_tasks, :]
        tumor_dist = output[0, self.n_tasks:, :]
        tissue_logits=[]
        tissue_maps=[]
        for idx, normal_tissue_prompt in enumerate(normal_dist):
            w_tissue= f.softmax(-normal_tissue_prompt, dim=-1)
            tissue_logits.append((tumor_dist[idx*2:(idx+1)*2] * w_tissue).sum(-1))
            tissue_maps.append(w_tissue)
        logits =torch.cat(tissue_logits).unsqueeze(0)
        self.text_similaritites =tissue_maps
        aggregation_logits = logits
        return aggregation_logits, inputs0

    def aggregate_no_textencoder(self,  inputs0, visual_projection=True):
            # get text features and normalize
        text_features= self.text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if visual_projection:
            #inputs0 = inputs0.view(-1, 768) @ self.visual_projection.T
            inputs0 = inputs0.view(-1, 512) @ self.visual_projection
        inputs0 = inputs0 / inputs0.norm(dim=-1, keepdim=True)

        # Class-Specific Region Feature Aggregation

        wandb.log({"logit_scale": self.logit_scale.exp().item()})
        output =self.logit_scale.exp()* f.conv1d(inputs0.unsqueeze(0).transpose(2, 1), text_features[:, :, None])
        normal_dist = output[0, :self.n_tasks, :]
        tumor_dist = output[0, self.n_tasks:, :]
        tissue_logits=[]
        tissue_maps=[]
        for idx, normal_tissue_prompt in enumerate(normal_dist):
            w_tissue= f.softmax(-normal_tissue_prompt, dim=-1)
            tissue_logits.append((tumor_dist[idx*2:(idx+1)*2] * w_tissue).sum(-1))
            tissue_maps.append(w_tissue)
        logits =torch.cat(tissue_logits).unsqueeze(0)
        self.text_similaritites =tissue_maps
        aggregation_logits = logits
        return aggregation_logits, inputs0

    @staticmethod
    def generate_task_mask(task=None, classes=8, n_classes_per_task=2):
        if task is None:
            return None
        task_mask = torch.zeros(classes)
        task_mask[task * n_classes_per_task:(task + 1) * n_classes_per_task] = 1
        return task_mask.cuda().bool()

    def get_attentions(self, images, cls_id=None, task_mask=None, table=None, add_context=True):
        _, new_inputs0, upsampled = upsample(images)
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        if self.args.dualcoop_multiscale:
            input = upsampled + new_inputs0
        else:
            input = new_inputs0
        cocoop_logits, proj = self.aggregate(prompts, tokenized_prompts, input)

        return self.text_similaritites, cocoop_logits

    def forward(self, images, cls_id=None, task_mask=None, table=None, add_context=None):
        # get image features
        x, new_inputs0, upsampled = upsample(images)
        if add_context is None:
            add_context = self.add_context
        else:
            self.add_context = add_context

        if not self.args.dsmil_multiscale:
            x = new_inputs0

        # train dsmil and get bag
        prediction_bag, Y_prob, Y_hat, A, classes, bag, task_meta_token = self.forward_dsmil_metanet(x)
        
        prompts, tokenized_prompts = self.prompt_learner(task_meta_token,task_mask,add_context)
        #prediction_bag,classes= torch.zeros(1, 8).cuda(), torch.zeros(10, 8).cuda()
        if self.args.dualcoop_multiscale:
            input = upsampled + new_inputs0
        else:
            input = new_inputs0
        #cocoop_logits, proj = self.aggregate(prompts, tokenized_prompts, input)
        #cocoop_logits= self.aggregate_no_textencoder(input)
        #return prediction_bag,classes,cocoop_logits,new_inputs0

        logits, _ = self.aggregate(prompts, tokenized_prompts, input)
        Y_hat = torch.argmax(logits)
        Y_prob = f.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, prediction_bag, classes, new_inputs0




    def get_keys(self):
        if len(self.old_keys) > 0:
            new = self.keys[len(self.old_keys):]
            keys = torch.cat([torch.stack(self.old_keys), new])
        else:
            keys = self.keys
        return keys

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params
