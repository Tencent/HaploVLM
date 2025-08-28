import torch
from diffusers.models.embeddings import CogVideoXPatchEmbed as CogVideoXPatchEmbed_
from diffusers.models.embeddings import get_3d_sincos_pos_embed


class CogVideoXPatchEmbed(CogVideoXPatchEmbed_):

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        pos_embedding = torch.from_numpy(pos_embedding).flatten(0, 1)
        joint_pos_embedding = torch.zeros(
            1, num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data.copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, input_embeds: torch.Tensor, visual = True):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
            visual : True for visual, False for text
        """
        if not visual:
            text_embeds = input_embeds
            text_embeds = self.text_proj(text_embeds)
            return text_embeds.contiguous()
        else:
            image_embeds = input_embeds
            batch, num_frames, channels, height, width = image_embeds.shape
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

            if self.use_positional_embeddings or self.use_learned_positional_embeddings:
                if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                    raise ValueError(
                        "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                        "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                    )

                pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

                if (
                    self.sample_height != height
                    or self.sample_width != width
                    or self.sample_frames != pre_time_compression_frames
                ):
                    pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
                    pos_embedding = pos_embedding.to(image_embeds.device, dtype=image_embeds.dtype)
                else:
                    pos_embedding = self.pos_embedding

                image_embeds = image_embeds + pos_embedding
            return image_embeds.contiguous()

        # embeds = torch.cat(
        #     [text_embeds, image_embeds], dim=1
        # ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        # return embeds
