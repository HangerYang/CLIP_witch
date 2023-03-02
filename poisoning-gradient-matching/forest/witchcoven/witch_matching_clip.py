"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchGradientMatchingClip(_Witch):

    def _batched_step(self, poison_delta, poison_delta_text, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current target loss."""
        images, token_ids, indices, attn_masks = example

        images = images.to(**self.setup)
        token_ids = token_ids.to(device=self.setup['device'], non_blocking=True)
        attn_masks = attn_masks.to(device=self.setup['device'], non_blocking=True)
        # Add adversarial pattern
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(indices.tolist()):
            lookup = kettle.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        images, token_ids, poison_slices, batch_positions, randgen = victim.distributed_control(
            images, token_ids, poison_slices, batch_positions)

        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            delta_slice_text = poison_delta_text[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()
            poison_images = images[batch_positions]
            images[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                images = kettle.augment(images, randgen=randgen)

            # Define the loss objective and compute gradients
            closure = self._define_objective(images, token_ids, attn_masks, delta_slice_text)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)
            delta_slice_text = victim.sync_gradients(delta_slice_text)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_delta_text.grad[poison_slices] = delta_slice_text.grad.detach().to(device=torch.device('cpu'))

                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective(self, images, token_ids, attn_masks, delta_slice_text):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, target_grad,  target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            clipOutput = model(input_ids=token_ids, attention_mask=attn_masks, pixel_values=images,
                                    text_delta=delta_slice_text)
            image_embeds = clipOutput.image_embeds  # normalize(clipOutput.image_embeds)
            text_embeds = clipOutput.text_embeds  # normalize(clipOutput.text_embeds)
            probs = torch.diagonal(image_embeds @ text_embeds.T)

            poison_loss = criterion(probs, torch.ones_like(probs))

            prediction = (probs > 0.5).sum()
            poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)

            passenger_loss = self._passenger_loss(poison_grad, target_grad, target_gnorm)
            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * poison_loss
            passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _passenger_loss(self, poison_grad, target_grad, target_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 5)
        else:
            indices = torch.arange(len(target_grad))

        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (target_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                poison_norm += poison_grad[i].pow(2).sum()

        if self.args.repel != 0:
            for i in indices:
                if self.args.loss in ['scalar_product', *SIM_TYPE]:
                    passenger_loss += self.args.repel * (target_grad[i] * poison_grad[i]).sum()
                elif self.args.loss == 'cosine1':
                    passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                elif self.args.loss == 'SE':
                    passenger_loss -= 0.5 * self.args.repel * (target_grad[i] - poison_grad[i]).pow(2).sum()
                elif self.args.loss == 'MSE':
                    passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

        passenger_loss = passenger_loss / target_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * poison_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / target_gnorm

        return passenger_loss

