import os
from abc import abstractmethod
from tqdm import tqdm
import time
import torch
import pandas as pd
from numpy import inf
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        self.model = model
        self.start_epoch = 1
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        

        if self.args.n_gpu > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus  # select multiple GPUs
            self.device = torch.device('cuda:0')  # always: 0
            self.model = self.model.to(self.device)
            print("GPUs_Used: {}".format(args.n_gpu))
            if args.resume is not None:  # the position is important!
                self._resume_checkpoint(args.resume)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus_id)  # always start with 0  # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" device_ids=[0, 1] 1 equals to GPU: 2
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu  # 0 1 2 3 select a uni-GPU
            self.device = torch.device('cuda:0')  # always: 0
            self.model = self.model.to(self.device)
            if args.resume is not None:
                self._resume_checkpoint(args.resume)  # the position is important!

        self.mnt_metric_val = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.checkpoint_dir = args.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists("".join([args.save_dir, "/logs"])):
            os.mkdir("".join([args.save_dir, "/logs"]))

        self.best_recorder = {'val': {self.mnt_metric_val: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}
        self.epochs_recorder = {}

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Resume training from epoch {}".format(self.start_epoch))
        self.model.load_state_dict(checkpoint['state_dict'])

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss_ = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            print("Epoch: {}".format(epoch))
            log = {'epoch': epoch}
            log.update(result)
            loss_.append(log['train_loss'])
            
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)

                except KeyError:
                    print("Warning: Metric '{}' is not found. " "performance monitoring is disabled.".format(
                        self.mnt_metric_test))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric_test]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Performance didn\'t improve for {} epochs. Stops.".format(self.early_stop))
                    break
            self._save_checkpoint(epoch, save_best=best)
            self.epochs_recorder.update(log)
            self._print_epochs_to_file()
            self._record_best(log)
            self._print_best_to_file()
            self._print_best()

    def _save_checkpoint(self, epoch, save_best=False):
        if self.args.n_gpu == 1:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best}

        elif self.args.n_gpu > 1:
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best
            }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

        if epoch % self.save_period == 0:
            file = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pth'.format(epoch))
            torch.save(state, file)
            print("Saving checkpoint: {} ...".format(file))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: {} ...".format(best_path))

    def _print_epochs_to_file(self):
        self.epochs_recorder['time'] = time.asctime(time.localtime(time.time()))
        self.epochs_recorder['visual_extractor'] = self.args.visual_extractor
        self.epochs_recorder['sample_method'] = self.args.sample_method
        self.epochs_recorder['seed'] = self.args.seed
        record_path = os.path.join(self.checkpoint_dir, self.args.dataset_name + '_epochs.csv')
        print("record_path : {}".format(record_path))
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        self.epochs_recorder["test_BO_M"] = self.epochs_recorder["test_METEOR"]
        self.epochs_recorder["test_BP_R"] = self.epochs_recorder["test_ROUGE_L"]
        self.epochs_recorder["test_BQ_C"] = self.epochs_recorder["test_CIDEr"]

        record_table = pd.concat([record_table, pd.DataFrame([self.epochs_recorder])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _print_best_to_file(self):
        record_path = os.path.join(self.checkpoint_dir, self.args.dataset_name + '_best.csv')
        print("record_path : {}".format(record_path))
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _record_best(self, log):
        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
            print("improved_test")

    def _print_best(self):
        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class TrainerOptimized(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(TrainerOptimized, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Training stage tracking
        self.text_encoder_frozen = False
        self.decoder_frozen = False
        self.current_stage = 1
        self.current_cycle = 1
        
        # Define stage boundaries for new strategy: Startup + 5 cycles of (1+1+1+1) epochs
        # Startup: 10 epochs of Text-to-Text training
        # Then 5 cycles of 4 epochs each = 20 epochs
        # Total epochs = 10 + 5 * 4 = 30 epochs
        self.startup_epochs = 10  # Startup phase: Text-to-Text training
        self.epochs_per_cycle = 4  # 1+1+1+1
        self.total_cycles = 5
        
        # Stage boundaries within each cycle
        self.stage_epochs = {
            1: 1,  # Stage 1: Image-to-Text (1 epoch)
            2: 1,  # Stage 2: Mix branch (1 epoch)
            3: 1,  # Stage 3: Traceback (1 epoch)
            4: 1   # Stage 4: Text-to-Text (1 epoch)
        }
        
        # For quick test mode
        if getattr(args, 'quick_test', False):
            self.startup_epochs = 2  # Reduced startup for quick test
            self.epochs_per_cycle = 4  # 1+1+1+1
            self.total_cycles = 1
            print("Quick test mode: 2 startup epochs + 1 cycle with 1 epoch per stage")
        else:
            print(f"Training strategy: {self.startup_epochs} startup epochs + {self.total_cycles} cycles, each with stages 1(1ep) + 2(1ep) + 3(1ep) + 4(1ep)")

    def _get_stage_from_epoch(self, epoch):
        """Determine current stage and cycle from epoch number"""
        # Convert to 0-based indexing
        epoch_0 = epoch - 1
        
        # Check if we're in startup phase
        if epoch_0 < self.startup_epochs:
            return 'startup', 0  # Special stage for startup
        
        # After startup phase, determine cycle and stage
        # Adjust epoch count after startup
        epoch_after_startup = epoch_0 - self.startup_epochs
        
        # Determine current cycle (0-based)
        cycle_0 = epoch_after_startup // self.epochs_per_cycle
        cycle = cycle_0 + 1  # Convert to 1-based
        
        # Determine position within current cycle (0-based)
        epoch_in_cycle_0 = epoch_after_startup % self.epochs_per_cycle
        epoch_in_cycle = epoch_in_cycle_0 + 1  # Convert to 1-based
        
        # Determine stage based on position within cycle
        if epoch_in_cycle <= self.stage_epochs[1]:
            stage = 1
        elif epoch_in_cycle <= self.stage_epochs[1] + self.stage_epochs[2]:
            stage = 2
        elif epoch_in_cycle <= self.stage_epochs[1] + self.stage_epochs[2] + self.stage_epochs[3]:
            stage = 3
        else:
            stage = 4
            
        return stage, cycle
    
    def _freeze_decoder(self):
        """Freeze the decoder"""
        if self.args.n_gpu > 1:
            for param in self.model.module.encoder_decoder.model.decoder.parameters():
                param.requires_grad = False
        else:
            for param in self.model.encoder_decoder.model.decoder.parameters():
                param.requires_grad = False
        self.decoder_frozen = True
        print("Decoder frozen")

    def _unfreeze_decoder(self):
        """Unfreeze the decoder"""
        if self.args.n_gpu > 1:
            for param in self.model.module.encoder_decoder.model.decoder.parameters():
                param.requires_grad = True
        else:
            for param in self.model.encoder_decoder.model.decoder.parameters():
                param.requires_grad = True
        self.decoder_frozen = False
        print("Decoder unfrozen")
    
    def _freeze_language_encoder(self):
        """Freeze the language encoder components"""
        if self.args.n_gpu > 1:
            for param in self.model.module.word_embd.parameters():
                param.requires_grad = False
            for param in self.model.module.att_embed_report.parameters():
                param.requires_grad = False
            for param in self.model.module.word_encoder.parameters():
                param.requires_grad = False
            for param in self.model.module.word_mlp.parameters():
                param.requires_grad = False
        else:
            for param in self.model.word_embd.parameters():
                param.requires_grad = False
            for param in self.model.att_embed_report.parameters():
                param.requires_grad = False
            for param in self.model.word_encoder.parameters():
                param.requires_grad = False
            for param in self.model.word_mlp.parameters():
                param.requires_grad = False
        self.text_encoder_frozen = True
        print("Language encoder frozen")

    def _unfreeze_language_encoder(self):
        """Unfreeze the language encoder components"""
        if self.args.n_gpu > 1:
            for param in self.model.module.att_embed_report.parameters():
                param.requires_grad = True
            for param in self.model.module.word_encoder.parameters():
                param.requires_grad = True
            for param in self.model.module.word_mlp.parameters():
                param.requires_grad = True
        else:
            for param in self.model.att_embed_report.parameters():
                param.requires_grad = True
            for param in self.model.word_encoder.parameters():
                param.requires_grad = True
            for param in self.model.word_mlp.parameters():
                param.requires_grad = True
        self.text_encoder_frozen = False
        print("Language encoder unfrozen")
    
    def _freeze_fusion_decoder(self):
        """Freeze the fusion decoder components (Memory, dictionary, w)"""
        if self.args.n_gpu > 1:
            for param in self.model.module.Memory.parameters():
                param.requires_grad = False
            for param in self.model.module.dictionary.parameters():
                param.requires_grad = False
            for param in self.model.module.w.parameters():
                param.requires_grad = False
        else:
            for param in self.model.Memory.parameters():
                param.requires_grad = False
            for param in self.model.dictionary.parameters():
                param.requires_grad = False
            for param in self.model.w.parameters():
                param.requires_grad = False
        print("Fusion decoder frozen")
    
    def _unfreeze_fusion_decoder(self):
        """Unfreeze the fusion decoder components (Memory, dictionary, w)"""
        if self.args.n_gpu > 1:
            for param in self.model.module.Memory.parameters():
                param.requires_grad = True
            for param in self.model.module.dictionary.parameters():
                param.requires_grad = True
            for param in self.model.module.w.parameters():
                param.requires_grad = True
        else:
            for param in self.model.Memory.parameters():
                param.requires_grad = True
            for param in self.model.dictionary.parameters():
                param.requires_grad = True
            for param in self.model.w.parameters():
                param.requires_grad = True
        print("Fusion decoder unfrozen")
    
    def _freeze_all_components(self):
        """Freeze all model components"""
        self._freeze_decoder()
        self._freeze_language_encoder()
        self._freeze_fusion_decoder()
        print("All components frozen")
    
    def _unfreeze_all_components(self):
        """Unfreeze all model components"""
        self._unfreeze_decoder()
        self._unfreeze_language_encoder()
        self._unfreeze_fusion_decoder()
        print("All components unfrozen")

    def _update_training_stage(self, epoch):
        """Update the training stage based on current epoch"""
        # Determine current stage and cycle
        new_stage, new_cycle = self._get_stage_from_epoch(epoch)
        prev_stage = self.current_stage
        prev_cycle = self.current_cycle
        
        self.current_stage = new_stage
        self.current_cycle = new_cycle
        
        # If stage or cycle changed, print message and update model parameters
        if self.current_stage != prev_stage or self.current_cycle != prev_cycle:
            if self.current_stage == 'startup':
                print(f"Startup Phase at epoch {epoch}")
                self._unfreeze_all_components()
                print("Startup: Training Text-to-Text, all components unfrozen")
            else:
                print(f"Cycle {self.current_cycle}, Stage {self.current_stage} at epoch {epoch}")
                
                # Stage 1: Image-to-Text (everything frozen)
                if self.current_stage == 1:
                    self._freeze_all_components()
                    print("Stage 1: Training Image-to-Text, all components frozen")
                
                # Stage 2: Mix branch (everything frozen)
                elif self.current_stage == 2:
                    self._freeze_all_components()
                    self._unfreeze_decoder()
                    print("Stage 2: Training Mix branch, all components frozen")
                
                # Stage 3: Traceback (everything frozen)
                elif self.current_stage == 3:
                    self._freeze_all_components()
                    self._unfreeze_decoder()
                    print("Stage 3: Training Traceback, all components frozen")
                
                # Stage 4: Text-to-Text (everything unfrozen)
                elif self.current_stage == 4:
                    self._unfreeze_all_components()
                    self._unfreeze_decoder()
                    print("Stage 4: Training Text-to-Text, all components unfrozen")

    def _train_epoch(self, epoch):
        # Update training stage based on current epoch
        self._update_training_stage(epoch)
        
        train_loss = 0
        self.model.train()
        
        # Create appropriate description for progress bar
        if self.current_stage == 'startup':
            desc = f'Epoch {epoch} - Startup Phase - Training'
        else:
            desc = f'Epoch {epoch} - Cycle {self.current_cycle} Stage {self.current_stage} - Training'
            
        with tqdm(desc=desc, unit='it', total=len(self.train_dataloader)) as pbar:
          
            for batch_idx, batch_data in enumerate(self.train_dataloader):
                # Handle both optimized and original dataloader formats
                if len(batch_data) == 9:  # Optimized format
                    (images_id, images, reports_ids, reports_masks, tok_ids, tok_masks, 
                     imageData, ark_embeddings, ark_predictions) = batch_data
                else:  # Original format
                    (images_id, images, reports_ids, reports_masks, tok_ids, tok_masks, imageData) = batch_data
                    ark_embeddings = ark_predictions = None
                
                # Move data to device
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                tok_ids = tok_ids.to(self.device)
                tok_masks = tok_masks.to(self.device)
                imageData = imageData.to(self.device)
                
                if ark_embeddings is not None:
                    ark_embeddings = ark_embeddings.to(self.device)
                if ark_predictions is not None:
                    ark_predictions = ark_predictions.to(self.device)
                
                reports_ids = tok_ids
                reports_masks = tok_masks
                
                # Startup Phase: Text-to-Text training
                if self.current_stage == 'startup':
                    output_t, output_v, emb_t_1, logit_t, logit_v = self.model(images, imageData, reports_ids, tok_ids, 
                                                             mode='train', tags=self.args.tags, epoch_id=epoch,
                                                             ark_embeddings=ark_embeddings, 
                                                             ark_predictions=ark_predictions,
                                                             stage=1)  # Use stage 1 for Text-to-Text
                    loss = self.criterion(output_t[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])
                    train_loss += loss.item()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                
                # Stage 1: Image-to-Text training (frozen)
                elif self.current_stage == 1:
                    output_t, output_v, emb_t_1, logit_t, logit_v = self.model(images, imageData, reports_ids, tok_ids, 
                                                             mode='train', tags=self.args.tags, epoch_id=epoch,
                                                             ark_embeddings=ark_embeddings, 
                                                             ark_predictions=ark_predictions,
                                                             stage=2)  # Use stage 2 for Image-to-Text
                    loss = self.criterion(output_v[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])
                    train_loss += loss.item()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                
                # Stage 2: Mix branch training (frozen)
                elif self.current_stage == 2:
                    output_mix, emb_t_1 = self.model(images, imageData, reports_ids, tok_ids, 
                                                     mode='train', tags=self.args.tags, epoch_id=epoch,
                                                     ark_embeddings=ark_embeddings, 
                                                     ark_predictions=ark_predictions,
                                                     stage=4)  # Use stage 4 for Mix branch
                    
                    # Mix branch loss
                    loss = self.criterion(output_mix[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])
                    
                    # Traceback consistency loss for Mix branch
                    # Get the argmax of the Mix branch output for traceback
                    with torch.no_grad():
                        sampleLogprobs, it = torch.max(output_mix[:, 9:, :].data, 2)
                        start_token = torch.zeros([it.shape[0], 1], dtype=it.dtype)
                        start_token = start_token.to(it.device)
                        it = torch.cat([start_token, it], dim=1)
                    
                    # Get traceback embedding for generated Mix output
                    _, emb_t = self.model(images, imageData, it, it, 
                                          mode='train', tags=self.args.tags, epoch_id=epoch,
                                          ark_embeddings=ark_embeddings, ark_predictions=ark_predictions,
                                          stage=4)  # Use stage 4 to get Mix branch output
                    train_loss += loss.item()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                
                # Stage 3: Traceback training (frozen)
                elif self.current_stage == 3:
                    output_t, output_v, emb_t_1, logit_t, logit_v = self.model(images, imageData, reports_ids, tok_ids, 
                                                                        mode='train', tags=self.args.tags, epoch_id=epoch,
                                                                        ark_embeddings=ark_embeddings, 
                                                                        ark_predictions=ark_predictions,
                                                                        stage=3)  # Use stage 3 for Traceback

                    # Calculate both text and image branch losses
                    loss_t = self.criterion(output_t[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])
                    loss_v = self.criterion(output_v[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])

                    temperature = 1.0
                    gumbel_output = F.gumbel_softmax(logit_v[:, 9:, :], tau=temperature, hard=True, dim=-1)  # [B, L, V], float, differentiable
                    start_token = torch.zeros([gumbel_output.shape[0], 1, gumbel_output.shape[-1]], device=gumbel_output.device)
                    start_token[:, 0, 0] = 1.0  # one-hot start token

                    it_onehot = torch.cat([start_token, gumbel_output], dim=1)  # [B, L+1, V]

                    embedding_weight = self.model.module.word_embd.weight
                    it_embedding = torch.matmul(it_onehot.float(), embedding_weight)  # [B, L+1, emb_dim]


                    emb_t, output_trace = self.model(images, imageData, it_embedding, reports_ids,
                                    mode='sample_cycle', tags=self.args.tags, epoch_id=epoch,
                                    ark_embeddings=ark_embeddings, ark_predictions=ark_predictions, stage=3)
                    
                    # MSE loss between generated embedding and ground truth embedding
                    loss_mse = self.mse_loss(emb_t, emb_t_1)
                    loss_trace_ce = self.criterion(output_trace[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])

                    # Total loss now includes text branch loss (loss_t)
                    loss = loss_trace_ce + loss_v # + loss_t
                    
                    train_loss += loss.item()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                
                # Stage 4: Text-to-Text training (unfrozen)
                elif self.current_stage == 4:
                    output_t, output_v, emb_t_1, logit_t, logit_v = self.model(images, imageData, reports_ids, tok_ids, 
                                                             mode='train', tags=self.args.tags, epoch_id=epoch,
                                                             ark_embeddings=ark_embeddings, 
                                                             ark_predictions=ark_predictions,
                                                             stage=1)  # Use stage 1 for Text-to-Text
                    loss = self.criterion(output_t[:, 9:, :], reports_ids[:, 1:], reports_masks[:, 1:])
                    train_loss += loss.item()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()

                pbar.set_postfix(loss=train_loss / (batch_idx + 1), 
                                cycle=self.current_cycle, 
                                stage=self.current_stage)
                pbar.update()

            # Avoid division by zero when train_dataloader is empty
            if len(self.train_dataloader) > 0:
                if self.current_stage == 'startup':
                    log = {'train_loss': train_loss / len(self.train_dataloader), 
                           'cycle': 0, 
                           'stage': 'startup'}
                else:
                    log = {'train_loss': train_loss / len(self.train_dataloader), 
                           'cycle': self.current_cycle, 
                           'stage': self.current_stage}
            else:
                if self.current_stage == 'startup':
                    log = {'train_loss': 0.0, 'cycle': 0, 'stage': 'startup'}
                else:
                    log = {'train_loss': 0.0, 'cycle': self.current_cycle, 'stage': self.current_stage}

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - Testing' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, batch_data in enumerate(self.test_dataloader):
                    # Handle both optimized and original dataloader formats
                    if len(batch_data) == 9:  # Optimized format
                        (images_id, images, reports_ids, reports_masks, tok_ids, tok_masks, 
                         imageData, ark_embeddings, ark_predictions) = batch_data
                    else:  # Original format
                        (images_id, images, reports_ids, reports_masks, tok_ids, tok_masks, imageData) = batch_data
                        ark_embeddings = ark_predictions = None
                    
                    # Move data to device
                    images = images.to(self.device)
                    reports_ids = reports_ids.to(self.device)
                    reports_masks = reports_masks.to(self.device)
                    tok_ids = tok_ids.to(self.device)
                    tok_masks = tok_masks.to(self.device)
                    imageData = imageData.to(self.device)
                    
                    if ark_embeddings is not None:
                        ark_embeddings = ark_embeddings.to(self.device)
                    if ark_predictions is not None:
                        ark_predictions = ark_predictions.to(self.device)
                    
                    # Choose sampling mode based on current stage
                    if self.current_stage == 'startup':
                        # Startup: Test with text branch
                        output = self.model(images, imageData, targets=None, tok=tok_ids, mode='sample_t', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    elif self.current_stage == 1:
                        # Stage 1: Test with image branch
                        output = self.model(images, imageData, targets=None, tok=None, mode='sample_v', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    elif self.current_stage == 2:
                        # Stage 2: Test with Mix branch
                        output = self.model(images, imageData, targets=None, tok=tok_ids, mode='sample_v', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    elif self.current_stage == 3:
                        # Stage 3: Test with image branch
                        output = self.model(images, imageData, targets=None, tok=None, mode='sample_v', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    elif self.current_stage == 4:
                        # Stage 4: Test with text branch
                        output = self.model(images, imageData, targets=None, tok=tok_ids, mode='sample_t', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    else:
                        # Default: Test with image branch
                        output = self.model(images, imageData, targets=None, tok=None, mode='sample_v', 
                                            tags=self.args.tags, epoch_id=epoch,
                                            ark_embeddings=ark_embeddings, ark_predictions=ark_predictions)
                    
                    if self.args.n_gpu > 1:
                        reports = self.model.module.tokenizer.decode_batch_tok(output.cpu().numpy())
                        ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        if self.args.tags ==1:
                            reports_tok = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
                            ground_truths_tok = self.model.module.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                    else:
                        reports = self.model.tokenizer.decode_batch_tok(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        if self.args.tags ==1:
                            reports_tok = self.model.tokenizer.decode_batch(output.cpu().numpy())
                            ground_truths_tok = self.model.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                    
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                    
                    # i = 0
                    # for id in images_id:
                    #     print(id)
                    #     print('Predicted Sent: {}'.format(reports[i]))
                    #     print('Reference Sent: {}'.format(ground_truths[i]))
                    #     print('\n')
                    #     if self.args.tags == 1:
                    #         print('Reference Sent.{}'.format(reports_tok[i]))
                    #         print('Reference Sent.{}'.format(ground_truths_tok[i]))
                    #     print('\n')
                    #     i = i + 1

                    i = 0
                    for id in images_id:
                        print(id)
                        print('Predicted Sent: {}'.format(reports[i]))
                        print('Reference Sent: {}'.format(ground_truths[i]))
                        print('\n')

                        print('Reference Sent_tok.{}'.format(reports_tok[i]))
                        print('Reference Sent_tok.{}'.format(ground_truths_tok[i]))
                        print('\n')
                        i = i + 1
                        
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print("test_", test_met)

            self.lr_scheduler.step()
        return log 