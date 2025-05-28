# builder.py

import torch
import torch.nn as nn
import torch.distributed as dist

from .resnet1d import CreateResNet1D

class CustomConv(nn.Module):
    def __init__(self):
        super(CustomConv, self).__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        # 가중치를 평균 필터로 초기화
        n = 4 * 1 * 1  # 입력 채널 수 * 커널 크기
        self.conv.weight.data.fill_(1.0 / n)

    def forward(self, x):
        return self.conv(x)

class MoCo(nn.Module):
    """
    Build a MoCo model with separate feature extraction and contrastive learning.
    Adapted for Distributed Data Parallel (DDP).
    """

    def __init__(self, dim=32, K=65536, m=0.999, T=0.07, mlp=True):
        """
        Args:
            base_encoder: the base encoder model (e.g., ResNet50)
            dim: feature dimension (default: 128)
            K: queue size; number of negative keys (default: 65536)
            m: momentum for updating key encoder (default: 0.999)
            T: softmax temperature (default: 0.07)
            mlp: whether to use a two-layer MLP projection head (default: True)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoders (query and key)
        self.encoder_q = CreateResNet1D(dim)
        self.encoder_k = CreateResNet1D(dim)

        self.final_conv = CustomConv()



        # Initialize the key encoder's parameters to match the query encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Not updated by gradient

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for the key encoder.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with the latest keys.
        """
        # When DDP is active, gather keys from all processes
        if dist.is_available() and dist.is_initialized():
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        K = self.queue.shape[1]
        assert K % batch_size == 0, "Queue size must be divisible by batch size"

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % K  # Move pointer

        self.queue_ptr[0] = ptr

    def extract_features_q(self, im_q):
        """
        Extract features from the query encoder.
        """
        # Compute query features
        q = self.encoder_q(im_q)  # Queries: B*32, 256
        q = nn.functional.normalize(q, dim=1)
        return q

    def extract_features_k(self, im_k):
        """
        Extract features from the key encoder.
        """
        with torch.no_grad():
            self._momentum_update_key_encoder()  # Update the key encoder

            # When DDP is active, perform batch shuffling
            if dist.is_available() and dist.is_initialized():
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k) 

            k = self.encoder_k(im_k)  #B*32, 256
            k = nn.functional.normalize(k, dim=1)

            # Undo batch shuffling
            if dist.is_available() and dist.is_initialized():
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        return k

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle for making use of BatchNorm.
        *** Only supports DistributedDataParallel (DDP) model. ***
        """
        # Gather from all GPUs
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = dist.get_world_size()

        # Random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # Broadcast to all GPUs
        dist.broadcast(idx_shuffle, src=0)

        # Index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # Shuffled index for this GPU
        rank = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[rank]

        x_shuffled = x_gather[idx_this]

        return x_shuffled, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only supports DistributedDataParallel (DDP) model. ***
        """
        # Gather from all GPUs
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = dist.get_world_size()

        # Restored index for this GPU
        rank = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[rank]

        x_unshuffled = x_gather[idx_this]

        return x_unshuffled

    def contrastive_loss(self, q, k):
        """
        Compute the contrastive loss using the extracted features.
        """
        # Positive logits: Nx1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1).detach()
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg_queue = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        batch_size = q.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long).cuda()
        mask = torch.eye(batch_size, dtype=torch.bool).cuda()
        l_neg_batch = torch.mm(q, k.t())
        l_neg_batch = l_neg_batch[~mask].view(batch_size, -1)


        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg_batch, l_neg_queue], dim=1)
        print(logits.shape)
        print(logits)

        # Apply temperature
        logits /= self.T

        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: A batch of query images
            im_k: A batch of key images
        Output:
            logits, labels
        """
        # Extract features
        q = self.extract_features_q(im_q)
        k = self.extract_features_k(im_k)

        # Compute contrastive loss
        logits, labels = self.contrastive_loss(q, k)

        return logits, labels

    def final(self, x):
        return self.final_conv(x)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensor.
    ***Warning***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor
