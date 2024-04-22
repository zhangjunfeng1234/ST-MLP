from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class PreNormResidual2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        return self.fn(x)


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def FeedForward2(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, 64),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size1,patch_size2, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.,output=9):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size1) == 0 and (image_w % patch_size2) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size1) * (image_w // patch_size2)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size1, p2 = patch_size2),
        nn.Linear((patch_size1 ** patch_size2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n (t d) -> b n t d',t=output,d=64),
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )


def MLPMixer2(*, image_size, channels, patch_size1,patch_size2, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size1) == 0 and (image_w % patch_size2) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size1) * (image_w // patch_size2)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size1, p2 = patch_size2),
        nn.Linear((patch_size1 ** patch_size2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n (t d) -> b n t d',t=12,d=64),
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )


def MLPMixer3(*, image_size, channels, patch_size1,patch_size2, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 2, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size1) == 0 and (image_w % patch_size2) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size1) * (image_w // patch_size2)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear


    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size1, p2 = patch_size2),
        nn.Linear((patch_size1 ** patch_size2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual2(dim, FeedForward2(dim, expansion_factor_token, dropout, chan_last))
        ),
        nn.Sequential(
            PreNormResidual(64, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(64, FeedForward(64, expansion_factor_token, dropout, chan_last))
        ),
        # nn.LayerNorm(64),
        Rearrange('b n (t d) -> b n t d',t=12,d=64),
        Reduce('b n t d -> b n d', 'mean'),
        # Repeat('b n d -> b n (t d)', t=1, d=64),
        # nn.Linear(dim, num_classes)
    )



def MLPMixer4(*, image_size, channels, patch_size1,patch_size2, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.,output):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size1) == 0 and (image_w % patch_size2) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size1) * (image_w // patch_size2)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size1, p2 = patch_size2),
        nn.Linear((patch_size1 ** patch_size2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n (t d) -> b n t d',t=output,d=128),
        # Reduce('b n t d -> b n d', 'mean'),

        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )



def MLPMixer0(*, image_size, channels, patch_size1,patch_size2, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size1) == 0 and (image_w % patch_size2) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size1) * (image_w // patch_size2)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size1, p2 = patch_size2),
        nn.Linear((patch_size1 ** patch_size2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n (t d) -> b n t d',t=35,d=2),
        # Reduce('b n t d -> b n d', 'mean'),

        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )