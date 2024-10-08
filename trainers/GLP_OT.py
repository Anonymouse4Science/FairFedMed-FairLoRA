import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from evaluation.metrics import compute_auc

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'GLP_OT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GLP_OT.N_CTX
        ctx_init = cfg.TRAINER.GLP_OT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.GLP_OT.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.GLP_OT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION


    def forward(self):
       
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        
        ctx = ctx.permute(1, 0, 2, 3) 
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.pixel_mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN)
        self.pixel_std = torch.tensor(self.cfg.INPUT.PIXEL_STD)

        self.n_cls = len(classnames)
        # Check if the dataset modality involves 3D input
        self.is_3d_input = cfg.DATASET.MODALITY_TYPE in {'oct_bscans', 'oct_bscans_3d'}
        if self.is_3d_input:
            self.dim_per_3d_slice = cfg.DATASET.DIM_PER_3D_SLICE 
            self.proj_per_3d_slice = nn.Conv2d(in_channels=self.dim_per_3d_slice, 
                                       out_channels=3, 
                                       kernel_size=5, 
                                       padding=2,
                                       dtype=clip_model.dtype)
            # Initialize the weights and biases
            std = self.dim_per_3d_slice ** -0.5
            nn.init.normal_(self.proj_per_3d_slice.weight, std=std)
            nn.init.zeros_(self.proj_per_3d_slice.bias)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.GLP_OT.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = cfg.TRAINER.GLP_OT.EPS
        self.max_iter = 100
        self.thresh = cfg.TRAINER.GLP_OT.THRESH
        self.OT = cfg.TRAINER.GLP_OT.OT
        self.top_percent = cfg.TRAINER.GLP_OT.TOP_PERCENT
        self.max_iter = cfg.TRAINER.GLP_OT.MAX_ITER

    def Sinkhorn(self, K, u, v):
        '''
        K is the Wasserstein distance, [bs*n_cls, 196, 77]
        u is , [bs*n_cls, 196]
        v is , [bs*n_cls, 77]
        '''
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        tmp = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2))
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version
        a is the source prob, [bs*n_cls, 196]
        b is the target prob, [bs*n_cls, 77]
        M is the cost matrix, i.e. Wasserstein distance, [bs*n_cls, 196, 77]

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {'err': []}
        stopThr=self.thresh 

        # K = torch.exp(M / (-reg))
        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while (cpt < numItermax):

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() <  stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u, dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v, dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

    def forward(self, image):
        b, c, h, w = image.shape
        if self.cfg.DATASET.NAME == "HarvardOph":
            image = image / 255.
            if self.is_3d_input:
                # split 3d input into multiple slices to process
                image = image.reshape(-1, self.dim_per_3d_slice, h, w)
                image = self.proj_per_3d_slice(image.type(self.dtype))

                # # Find the minimum and maximum values per batch
                min_vals = image.amin(dim=(1, 2, 3), keepdim=True)
                max_vals = image.amax(dim=(1, 2, 3), keepdim=True)
                # Normalize to range [0, 1]
                image = (image - min_vals) / (max_vals - min_vals + 1e-5)  

            image = image - self.pixel_mean.reshape(1,-1,1,1).to(image.device)
            image = image / self.pixel_std.reshape(1,-1,1,1).to(image.device)

        image_features = self.image_encoder(image.type(self.dtype), attr=None)  
        image_feature_pool = image_features[0]
        image_features = image_features[1:]  
        M = image_features.shape[0]  # 14*14
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()   
        tokenized_prompts = self.tokenized_prompts
        if self.dataset == "ImageNet":
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1)) 
            text_features = text_features.to(self.device)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts) 
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        
        image_features =  F.normalize(image_features, dim=2) 
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M, self.N, -1)  # num_pixels, 2,  batch_size * n_cls
        sim = sim.permute(2,0,1)       # batch_size * n_cls, num_pixels, 2
        wdist = 1.0 - sim

        xx = torch.zeros(sim.shape[0], M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        if self.OT == 'Sinkhorn':
            yy = torch.zeros(sim.shape[0], self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        elif self.OT == 'COT':
            top_percent = min(torch.sum(xx).item(), self.top_percent)
            yy = torch.zeros(sim.shape[0], self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N) * top_percent
        elif self.OT == 'None':
            pass
        else:
            raise NotImplementedError

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == 'Sinkhorn':
                T = self.Sinkhorn(KK, xx, yy)  # T is the transport plan
                if torch.isnan(T).any():
                    return None
            elif self.OT == 'COT':
                T = self.entropic_COT_fast(xx, yy, KK,0.01,numItermax=self.max_iter)
                if torch.isnan(T).any():
                    return None
            elif self.OT == 'None':
                T = 1
            else:
                raise NotImplementedError

        if self.OT == 'None':
            sim_op = torch.mean(T * sim, dim=(1, 2))
        else:
            sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, -1, self.n_cls)
        sim_op = sim_op.mean(1)  # average all slices 
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op   
        
        return logits


# @TRAINER_REGISTRY.register()
class GLP_OT(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GLP_OT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.pixel_mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN)
        self.pixel_std = torch.tensor(self.cfg.INPUT.PIXEL_STD)

        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GLP_OT.PREC == "fp32" or cfg.TRAINER.GLP_OT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()   

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER:
                # only unfreeze ln_pre and ln_post
                if name.startswith('image_encoder.ln_pre'):
                    param.requires_grad_(True)
                    continue
            
            if cfg.TRAINER.GLP_OT_LORA.UNFREEZE_TEXT_ENCODER:
                if name.startswith('text_encoder.ln_'):
                    param.requires_grad_(True)
                    continue
            
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME== "ImageNet":
            self.device =  torch.device("cuda:0")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        params_to_optimize = list(self.model.prompt_learner.parameters())
        if self.model.is_3d_input:
            params_to_optimize += list(self.model.proj_per_3d_slice.parameters())
        if cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER:
            params_to_optimize += list(self.model.image_encoder.parameters())
            self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)
        else:
            # NOTE: only give prompt_learner to the optimizer
            self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # Register the prompt learner
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        if cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER:
            # Register the image encoder
            self.register_model("image_encoder", self.model.image_encoder, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GLP_OT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch, is_last_client=False):
        image, label = self.parse_batch_train(batch)[:2]
        
        prec = self.cfg.TRAINER.GLP_OT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        if output.shape == label.shape:
            output_prob = output.sigmoid()
        else:
            output_prob = output.softmax(-1)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if self.cfg.DATASET.NAME == "HarvardOph":
            loss_summary["auc"] = compute_auc(output_prob, label).item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        if self.cfg.DATASET.NAME == "HarvardOph":
            # input = input / 255.
            # input = input - self.pixel_mean.reshape(1,-1,1,1).to(input.device)
            # input = input / self.pixel_std.reshape(1,-1,1,1).to(input.device)

            attrs = batch["attrs"].t()
            return input, label, attrs
        else:
            return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        if self.cfg.DATASET.NAME == "HarvardOph":
            # input = input / 255.
            # input = input - self.pixel_mean.reshape(1,-1,1,1).to(input.device)
            # input = input / self.pixel_std.reshape(1,-1,1,1).to(input.device)

            attrs = batch["attrs"].t()
            return input, label, attrs
        else:
            return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)