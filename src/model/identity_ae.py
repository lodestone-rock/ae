import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from einops import rearrange
import torch._dynamo as dynamo

dynamo.config.cache_size_limit = 64


def soft_clamp(x, scale, alpha, shift):
    return scale[None, :, None, None] * F.tanh(x * alpha) + shift[None, :, None, None]


class SoftClamp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.use_compiled = False

    def forward(self, x):
        if self.use_compiled:
            return torch.compile(soft_clamp)(x, self.scale, self.alpha, self.shift)
        else:
            return soft_clamp(x, self.scale, self.alpha, self.shift)


# @torch.compile
# def pointwise_gating(x, act_fn):
#     lin, gate = rearrange(x, "n (g c) h w -> g n c h w", g=2)
#     return lin * act_fn(gate)

def pointwise_gating(x, act_fn):
    # x shape: (n, 2*c, h, w)
    # Split along channel dimension into two equal parts
    lin, gate = torch.chunk(x, chunks=2, dim=1)
    return lin * act_fn(gate)


class SimpleResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        act_fn=torch.sin,
    ):
        super(SimpleResNetBlock, self).__init__()
        self.rms_norm = SoftClamp(out_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=2
        )
        self.pointwise = nn.Conv2d(in_channels // 2, out_channels, 1, stride, padding)
        self.act_fn = act_fn
        torch.nn.init.zeros_(self.pointwise.weight)
        torch.nn.init.zeros_(self.pointwise.bias)

    def forward(self, x, cond=None):
        skip = x
        x = self.rms_norm(x)
        
        # add conditional stuff as a simple addition
        if cond is not None:
            x = cond + x
        x = self.conv(x)
        x = pointwise_gating(x, self.act_fn)
        # fused using torch compile
        # lin, gate = rearrange(x, "n (g c) h w -> g n c h w", g=2)
        # x = lin * self.act_fn(gate)
        x = self.pointwise(x)
        return x + skip


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownBlock, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels * (scale_factor**2), out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.pointwise_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels // (scale_factor**2),
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.pointwise_conv(x)
        return x


class DownBlockNonParametric(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.pixel_unshuffle = nn.PixelUnshuffle(scale_factor)

        # Calculate the number of channels after pixel unshuffle
        intermediate_channels = in_channels * (scale_factor**2)

        self.num_channels_per_group = intermediate_channels // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, C_in, H, W)
        x = self.pixel_unshuffle(x)
        # x: (B, C_in * sf^2, H/sf, W/sf) = (B, C_intermediate, H', W')

        # Reshape to group channels for averaging
        # Shape becomes (B, C_out, channels_per_group, H', W')
        b, c_inter, h_prime, w_prime = x.shape
        x = x.view(b, self.out_channels, self.num_channels_per_group, h_prime, w_prime)

        # Average across the grouped channels dimension (dim=2)
        x = torch.mean(x, dim=2)
        # x: (B, C_out, H', W') = (B, C_out, H/sf, W/sf)

        return x


class UpBlockNonParametric(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # Calculate the number of channels after pixel shuffle
        intermediate_channels = out_channels * (scale_factor**2)

        self.repeat_factor = intermediate_channels // in_channels
        print()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Repeat channels using repeat_interleave along the channel dimension (dim=1)
        x = torch.repeat_interleave(x, repeats=self.repeat_factor, dim=1)
        # x: (B, C_in, H, W)
        # x: (B, C_in / sf^2, H*sf, W*sf)
        x = self.pixel_shuffle(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        down_layer_blocks=((32, 2), (64, 2), (128, 2)),
        act_fn=torch.sin,
    ):
        super(Encoder, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels,
            down_layer_blocks[0][0],
            kernel_size=1,
            stride=1,
            padding="same",
        )

        down_blocks = nn.ModuleList()
        res_blocks = nn.ModuleList()
        for i, blocks in enumerate(down_layer_blocks):
            # (64, 2) dim, block count
            res_block = nn.ModuleList()
            # first down block skipped
            if i != 0:
                downsample = DownBlockNonParametric(
                    down_layer_blocks[i - 1][0], down_layer_blocks[i][0]
                )
                down_blocks.append(downsample)
            for block in range(blocks[1]):
                # (2) block count
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same", act_fn)
                res_block.append(resnet)
            res_blocks.append(res_block)

        self.down_blocks = down_blocks
        self.res_blocks = res_blocks

        self.out_norm = SoftClamp(down_layer_blocks[-1][0])
        self.out_conv = nn.Conv2d(
            down_layer_blocks[-1][0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        nn.init.zeros_(self.in_conv.bias)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, checkpoint=True, skip_last_downscale=False):
        x = self.in_conv(x)
        res_blocks_count = len(self.res_blocks)

        for i, res_blocks in enumerate(self.res_blocks):
            is_last = i == res_blocks_count - 1
            if i != 0:
                if not (skip_last_downscale and is_last):
                    x = self.down_blocks[i - 1](x)

            for resnet in res_blocks:
                if checkpoint:
                    x = ckpt.checkpoint(resnet, x)
                else:
                    x = resnet(x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up_layer_blocks=((128, 2), (64, 2), (32, 2)),
        act_fn=torch.sin,
    ):
        super(Decoder, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels, up_layer_blocks[0][0], kernel_size=1, stride=1, padding="same"
        )

        up_blocks = nn.ModuleList()
        res_blocks = nn.ModuleList()
        for i, blocks in enumerate(up_layer_blocks):
            # (64, 2) dim, block count
            res_block = nn.ModuleList()
            # first up block skipped
            if i != 0:
                upsample = UpBlockNonParametric(
                    up_layer_blocks[i - 1][0], up_layer_blocks[i][0]
                )
                up_blocks.append(upsample)
            for block in range(blocks[1]):
                # (2) block count
                resnet = SimpleResNetBlock(blocks[0], blocks[0], 3, 1, "same", act_fn)
                res_block.append(resnet)
            res_blocks.append(res_block)

        self.up_blocks = up_blocks
        self.res_blocks = res_blocks

        self.out_norm = SoftClamp(up_layer_blocks[-1][0])
        self.out_conv = nn.Conv2d(
            up_layer_blocks[-1][0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        nn.init.zeros_(self.in_conv.bias)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, checkpoint=True, skip_second_upscale=False, logits_only=False):
        x = self.in_conv(x)

        for i, res_blocks in enumerate(self.res_blocks):
            if i != 0:
                is_second = i == 1
                if not (skip_second_upscale and is_second):
                    x = self.up_blocks[i - 1](x)

            for resnet in res_blocks:
                if checkpoint:
                    x = ckpt.checkpoint(resnet, x)
                else:
                    x = resnet(x)

        if not logits_only:
            x = self.out_norm(x)
            x = self.out_conv(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(
        self,
        pixel_channels=3,
        bottleneck_channels=4,
        up_layer_blocks=((32, 2), (64, 2), (128, 2)),
        down_layer_blocks=((32, 2), (64, 2), (128, 2)),
        act_fn="sin",
        **kwargs
    ):
        super(AutoEncoder, self).__init__()

        activation_functions = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "elu": F.elu,
            "selu": F.selu,
            "gelu": F.gelu,
            "silu": F.silu,
            "sin": torch.sin,
        }

        self.encoder = Encoder(
            pixel_channels,
            bottleneck_channels,
            down_layer_blocks,
            activation_functions[act_fn],
        )
        self.decoder = Decoder(
            bottleneck_channels,
            pixel_channels,
            up_layer_blocks,
            activation_functions[act_fn],
        )

    def encode(self, x, checkpoint=True, skip_last_downscale=False):
        return self.encoder(x, checkpoint, skip_last_downscale)

    def decode(self, x, checkpoint=True, skip_second_upscale=False):
        return self.decoder(x, checkpoint, skip_second_upscale)

    def forward(self, x, checkpoint=True):
        x = self.encode(x, checkpoint)
        return self.decode(x, checkpoint)


class CEAutoEncoder(nn.Module):
    def __init__(
        self,
        pixel_channels=3,
        bottleneck_channels=4,
        up_layer_blocks=((32, 2), (64, 2), (128, 2)),
        down_layer_blocks=((32, 2), (64, 2), (128, 2)),
        act_fn="sin",
        **kwargs
    ):
        super(CEAutoEncoder, self).__init__()

        activation_functions = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "elu": F.elu,
            "selu": F.selu,
            "gelu": F.gelu,
            "silu": F.silu,
            "sin": torch.sin,
        }

        self.encoder = Encoder(
            pixel_channels,
            bottleneck_channels,
            down_layer_blocks,
            activation_functions[act_fn],
        )
        self.decoder = Decoder(
            bottleneck_channels,
            pixel_channels*256,
            up_layer_blocks,
            activation_functions[act_fn],
        )

    def encode(self, x, checkpoint=True, skip_last_downscale=False):
        return self.encoder(x, checkpoint, skip_last_downscale)

    def decode(self, x, checkpoint=True, skip_second_upscale=False):
        return self.decoder(x, checkpoint, skip_second_upscale)

    def forward(self, x, checkpoint=True):
        x = self.encode(x, checkpoint)
        return self.decode(x, checkpoint)


class FlowmatchingHead(nn.Module):
    def __init__(
        self,
        pixel_channels,
        layer_blocks=(128, 2),
        act_fn=torch.sin,
    ):
        super(FlowmatchingHead, self).__init__()
        self.in_conv = nn.Conv2d(
            pixel_channels, layer_blocks[0], kernel_size=1, stride=1, padding="same"
        )

        # residual blocks
        res_blocks = nn.ModuleList()
        for block in range(layer_blocks[1]):
            resnet = SimpleResNetBlock(layer_blocks[0], layer_blocks[0], 3, 1, "same", act_fn)
            res_blocks.append(resnet)
        self.res_blocks = res_blocks

        self.out_norm = SoftClamp(layer_blocks[0])
        self.out_conv = nn.Conv2d(
            layer_blocks[0],
            pixel_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        nn.init.zeros_(self.in_conv.bias)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, cond, checkpoint=True):
        x = self.in_conv(x)

        for resnet in self.res_blocks:
            if checkpoint:
                x = ckpt.checkpoint(resnet, x, cond)
            else:
                x = resnet(x, cond)

        x = self.out_norm(x)
        x = self.out_conv(x)

        return x
    

class FlowmatchingAutoEncoder(nn.Module):
    def __init__(
        self,
        pixel_channels=3,
        bottleneck_channels=4,
        up_layer_blocks=((32, 2), (64, 2), (128, 2)),
        down_layer_blocks=((32, 2), (64, 2), (128, 2)),
        flow_matching_head_block=(32, 2),
        act_fn="sin",
        **kwargs
    ):
        super(FlowmatchingAutoEncoder, self).__init__()

        activation_functions = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "elu": F.elu,
            "selu": F.selu,
            "gelu": F.gelu,
            "silu": F.silu,
            "sin": torch.sin,
        }

        self.encoder = Encoder(
            pixel_channels,
            bottleneck_channels,
            down_layer_blocks,
            activation_functions[act_fn],
        )
        self.decoder = Decoder(
            bottleneck_channels,
            pixel_channels,
            up_layer_blocks,
            activation_functions[act_fn],
        )

        # projection to match the activation channel size with the flow matching head
        self.flowmatch_proj = nn.Conv2d(
            up_layer_blocks[-1][0],
            flow_matching_head_block[0],
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.flow_matching_head = FlowmatchingHead(
            pixel_channels,
            flow_matching_head_block,
            activation_functions[act_fn],            
        )


    # put this euler ODE solver as method here for now
    @staticmethod
    def denoise_cfg(
        model,
        timesteps: list[float],
        img,
        cond,
        neg_cond=None,
        cfg: float = 2.0,
        first_n_steps_without_cfg: int = 0,
    ):
        # neg cond can be zero tensor 
        if neg_cond is None:
            neg_cond = torch.zeros_like(cond)
        step_count = 0
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = model(
                x=img,
                guidance_logits=cond,
            )
            # disable cfg for x steps before using cfg
            if step_count < first_n_steps_without_cfg or first_n_steps_without_cfg == -1:
                img = img.to(pred) + (t_prev - t_curr) * pred
            else:
                pred_neg = model(
                    x=img,
                    guidance_logits=neg_cond, 
                )

                pred_cfg = pred_neg + (pred - pred_neg) * cfg

                img = img + (t_prev - t_curr) * pred_cfg

            step_count += 1

        return img

    @staticmethod
    def create_distribution(num_points, device=None):
        # Probability range on x axis
        x = torch.linspace(0, 1, num_points, device=device)

        # Custom probability density function
        probabilities = -7.7 * ((x - 0.5) ** 2) + 2

        # Normalize to sum to 1
        probabilities /= probabilities.sum()

        return x, probabilities

    @staticmethod
    def sample_from_distribution(x, probabilities, num_samples, device=None):
        # Step 1: Compute the cumulative distribution function
        cdf = torch.cumsum(probabilities, dim=0)

        # Step 2: Generate uniform random samples
        uniform_samples = torch.rand(num_samples, device=device)

        # Step 3: Map uniform samples to the x values using the CDF
        indices = torch.searchsorted(cdf, uniform_samples, right=True)

        # Get the corresponding x values for the sampled indices
        sampled_values = x[indices]

        return sampled_values
    
    @staticmethod
    def fm_target(noise, image):
        return noise - image

    def encode(self, x, checkpoint=True, skip_last_downscale=False):
        return self.encoder(x, checkpoint, skip_last_downscale)

    def decode(self, x, checkpoint=True, skip_second_upscale=False, logits_only=True):
        # use logits only to guide flow matching head
        return self.decoder(x, checkpoint, skip_second_upscale, logits_only)
    
    def flow_matching_decode(self, x, guidance_logits):
        # x is noise and the prev activations act like a guidance embedding
        # this returns image vectors for ODE process
        guidance_logits = self.flowmatch_proj(guidance_logits)
        return self.flow_matching_head(x, guidance_logits)

