import torch 
import torch.nn as nn
from typing import Tuple
from .modules import ImagingMaskedEncoder, ImagingMaskedDecoder, ReconstructionCriterion
from .utils.imaging_model_related import Masker, sincos_pos_embed, patchify, unpatchify
from medlat.utils import init_from_ckpt
from medlat.registry import register_model



@register_model("token.vita.reconmae")
class ReconMAE(nn.Module):
    def __init__(self, 
                img_shape: tuple,
                patch_embed_cls: str = "PatchEmbed",
                patch_size: Tuple[int, ...] = (5, 8, 8),
                patch_in_channels: int = 1,
                pixel_unshuffle_scale: int = 1,
                mask_type: str = "random",
                mask_ratio: float = 0.0,
                circular_pe: bool = False,
                use_enc_pe: bool = True,
                mask_loss: bool = True,
                shift_size: Tuple[int, ...] = (0, 0, 0),
                enc_embed_dim: int = 1025,
                enc_depth: int = 6,
                enc_num_heads: int = 5,
                mlp_ratio: float = 4.0,
                grad_checkpointing: bool = False,
                use_both_axes: bool = False,
                dec_embed_dim: int = 1025,
                dec_num_heads: int = 5,
                dec_depth: int = 2,
                ckpt_path: str = None,
                loss_types: Tuple[str, ...] = ("mse"),
                loss_weights: Tuple[float, ...] = (1.0),
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.data_view = self.hparams.val_dset.view
        # use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        # img_shape = self.hparams.val_dset[0][0].shape
        print(f"img_shape: {img_shape}, use_both_axes: {use_both_axes}, mask_ratio: {mask_ratio}")
        self.patch_size = patch_size
        self.img_shape = img_shape

        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, 
                                                    use_both_axes=use_both_axes,
                                                    patch_embed_cls=patch_embed_cls,
                                                    patch_size=patch_size,
                                                    patch_in_channels=patch_in_channels,
                                                    pixel_unshuffle_scale=pixel_unshuffle_scale,
                                                    mask_type=mask_type,
                                                    mask_ratio=mask_ratio,
                                                    circular_pe=circular_pe,
                                                    use_enc_pe=use_enc_pe,
                                                    mask_loss=mask_loss,
                                                    shift_size=shift_size,
                                                    enc_embed_dim=enc_embed_dim,
                                                    enc_depth=enc_depth,
                                                    enc_num_heads=enc_num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    grad_checkpointing=grad_checkpointing,
                                                    )
        self.decoder_imaging = ImagingMaskedDecoder(decoder_num_patches=self.encoder_imaging.patch_embed.num_patches, 
                                                    grid_size=self.encoder_imaging.patch_embed.grid_size,                               
                                                    head_out_dim=self.encoder_imaging.patch_p_num,
                                                    dec_num_heads=dec_num_heads,
                                                    dec_depth=dec_depth,
                                                    mlp_ratio=mlp_ratio,
                                                    use_enc_pe=use_enc_pe,
                                                    enc_embed_dim=enc_embed_dim,
                                                    dec_embed_dim=dec_embed_dim,
                                                    use_both_axes=use_both_axes,
                                                    )
        self.reconstruction_criterion = ReconstructionCriterion(loss_types=loss_types, loss_weights=loss_weights)

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)
        
    def forward(self, x):
        x, mask, ids_restore = self.encoder_imaging(x)
        x = self.decoder_imaging(x, ids_restore)
        b = x.shape[0]
        img_shape = (b, *self.img_shape)
        x = unpatchify(x, im_shape=img_shape, patch_size=self.patch_size)
        return x, mask
        
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, sub_idx = batch

        pred_patches, mask = self.forward(imgs)

        imgs_patches = patchify(imgs, patch_size=self.hparams.patch_size)
        loss_dict = self.reconstruction_criterion(pred_patches, imgs_patches, mask) 
        psnr_value = calculate_psnr(pred_patches, imgs_patches, mask, replace_with_gt=True)
        
        # Logging metrics and median
        self.log_recon_metrics(loss_dict, psnr_value, mode=mode)

        if mode == "val": # For checkpoint tracking
            self.module_logger.update_metric_item(f"{mode}_PSNR", psnr_value, mode=mode) 
            
        log_rate = eval(f"self.hparams.{mode}_log_rate")
        if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
            if (sub_idx == 0).any():
                i = (sub_idx == 0).argwhere().squeeze().item()
                self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            if (sub_idx == 1).any():
                i = (sub_idx == 1).argwhere().squeeze().item()
                self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode=mode)
        return loss_dict["loss"]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, sub_idx = batch

        pred_patches, mask = self.forward(imgs)

        pred_imgs = unpatchify(pred_patches, im_shape=imgs.shape, patch_size=self.hparams.patch_size)
        psnr_value = calculate_psnr(pred_imgs, imgs, reduction="none") # (B, S)
        psnr_value = np.mean(psnr_value, axis=2)
        self.test_psnr.append(psnr_value)

        # Save sample images
        if (sub_idx == 0).any():
            i = (sub_idx == 0).argwhere().squeeze().item()
            sample_imgs = imgs[i]
            sample_recon_imgs = pred_imgs[i]
            self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode="test")

            if self.hparams.test_sample_path is not None:
                Path(self.hparams.test_sample_path).mkdir(parents=True, exist_ok=True)
                view_name = self.hparams.test_sample_path.split('_')[-1]
                sample_imgs = sample_imgs.detach().cpu().numpy()
                sample_recon_imgs = sample_recon_imgs.detach().cpu().numpy()
                for k in range(sample_imgs.shape[0]):
                    gt = sample_imgs[k, 13]
                    recon = sample_recon_imgs[k, 13]
                    plt.imsave(Path(self.hparams.test_sample_path) / f"gt_0_s{k}_t13.png", gt, cmap="gray")
                    plt.imsave(Path(self.hparams.test_sample_path) / f"recon_{view_name}_0_s{k}_t13.png", recon, cmap="gray")

    def on_test_epoch_end(self) -> None:
        test_psnr = np.concatenate(self.test_psnr, axis=0) # (#test, S)
        psnr_mean = np.mean(test_psnr)
        psnr_std = np.std(test_psnr)
        # Short-axis
        if self.data_view == 0:
            psnr_sax_mean = psnr_mean
            psnr_sax_std = psnr_std
            psnr_lax_mean = 0
            psnr_lax_std = 0
        # Long-axis
        elif self.data_view == 1:
            psnr_sax_mean = 0
            psnr_sax_std = 0
            psnr_lax_mean = psnr_mean
            psnr_lax_std = psnr_std
        elif self.data_view == 2:
            psnr_lax_mean = np.mean(test_psnr[:, :3])
            psnr_lax_std = np.std(test_psnr[:, :3])
            psnr_sax_mean = np.mean(test_psnr[:, 3:])
            psnr_sax_std = np.std(test_psnr[:, 3:])
        results = {"psnr_mean": psnr_mean, "psnr_std": psnr_std, 
                   "psnr_sax_mean": psnr_sax_mean, "psnr_sax_std": psnr_sax_std, 
                   "psnr_lax_mean": psnr_lax_mean, "psnr_lax_std": psnr_lax_std, }
        
        # Table logging
        columns = list(results.keys())
        test_table = wandb.Table(columns=columns)
        test_table.add_data(*[results[col] for col in columns])
        wandb.log({"Evaluation_table": test_table})

        if self.hparams.test_psnr_path is not None:
            Path(self.hparams.test_psnr_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.hparams.test_psnr_path, "wb") as file:
                pickle.dump(results, file)

        self.wandb_log(self.current_epoch, mode="test")
                
    @torch.no_grad()
    def generate_latents(self, data_loader, 
                         token_path: str, 
                         tsne_map_path: str,
                         save_all_patch_tokens: bool = True,
                         save_tsne: bool = True,
                         ):
        latents = []
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating latent codes"):
            imgs, sub_idx = batch
            if torch.cuda.is_available():
                imgs = imgs.to("cuda")
            sub_path = data_loader.dataset.subject_paths[sub_idx]
            sub_id = sub_path.parent.name
            enc_output_latent, _, ids_restore = self.encoder_imaging(imgs)
            cls_t = enc_output_latent[:, 0, :]
            all_t_ = enc_output_latent[:, 1:, :]
            
            # Restore the order of the tokens
            all_t_restore = torch.gather(all_t_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, all_t_.shape[-1]))
            
            # Take the average of tokens across spatial dimensions
            B, S, T = imgs.shape[:3]
            num_t_patches = T // self.hparams.patch_size[0]
            all_t_restore = all_t_restore.reshape(B, S, num_t_patches, -1, all_t_restore.shape[-1])
            all_t_restore = all_t_restore.moveaxis(1, 2)
            all_t = all_t_restore.reshape(B, num_t_patches, -1, all_t_restore.shape[-1]).mean(2)
            
            if save_all_patch_tokens:
                latents.append({'subj_id': np.array(int(sub_id)).reshape(1, 1), 
                                'cls_token': cls_t.detach().cpu().numpy(), 
                                'all_token': all_t.detach().cpu().numpy()})
            else:
                latents.append({'subj_id': np.array(int(sub_id)).reshape(1, 1), 
                                'cls_token': cls_t.detach().cpu().numpy(),})
        
        # Save the latent codes
        if not save_all_patch_tokens:
            cls_token = np.concatenate([i['cls_token'] for i in latents])
            subj_id = np.concatenate([i['subj_id'] for i in latents]).reshape(-1)
            np.savez(token_path, cls_token=cls_token, subj_id=subj_id)
        else:
            cls_token = np.concatenate([i['cls_token'] for i in latents])
            subj_id = np.concatenate([i['subj_id'] for i in latents]).reshape(-1)
            all_token = np.concatenate([i['all_token'] for i in latents])
            np.savez(token_path, cls_token=cls_token, all_token=all_token, subj_id=subj_id)
        
        if save_tsne:
        # Save the t-SNE embeddings
            scaled_avg = StandardScaler().fit_transform(np.mean(all_token, axis=1))
            scaled_cls = StandardScaler().fit_transform(cls_token)
            scaled_tmp = StandardScaler().fit_transform(all_token.reshape(-1, all_token.shape[2]))
            
            tsne_map_cls = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_cls)
            tsne_map_avg = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_avg)
            tsne_map_tmp = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_tmp)
                
            np.savez(tsne_map_path, 
                    tsne_map_cls=tsne_map_cls, 
                    tsne_map_avg=tsne_map_avg, 
                    tsne_map_tmp=tsne_map_tmp, 
                    subj_id=subj_id)
