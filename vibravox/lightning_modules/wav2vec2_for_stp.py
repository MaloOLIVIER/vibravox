from functools import partial
from typing import Any, Dict, List

import torch
import transformers
import logging
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2Processor

logger: logging.Logger = logging.getLogger(__name__)

class Wav2Vec2ForSTPLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC,
        optimizer: partial[torch.optim.Optimizer],
        scheduler: partial[torch.optim.lr_scheduler],
        unfreeze_at_step: int,
        push_to_hub_after_testing: bool = False,
        nb_steps_per_epoch: int = 1,
        max_epochs: int = 1,
        description: str = None,
    ):
        """
        Definition of Wav2Vec2ForSTP and its training pipeline with pytorch lightning paradigm

        Args:

            wav2vec2_for_ctc (torch.nn.Module): Neural network to enhance the speech
            optimizer (partial[torch.optim.Optimizer]): Optimizer


            push_to_hub_after_testing (bool): If True, the model is pushed to the Hugging Face hub after testing. Defaults to False.
            description (str): Description to log in tensorboard
        """
        super().__init__()

        self.sample_rate: int = sample_rate
        self.wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC = wav2vec2_for_ctc(
            pad_token_id=36,  # Corresponds to `self.trainer.datamodule.tokenizer.pad_token_id`
            vocab_size=39,  # Corresponds to `len(self.trainer.datamodule.tokenizer)`
        )

        self.wav2vec2_for_ctc.freeze_feature_extractor()

        self.optimizer_factory: torch.optim.Optimizer = optimizer(params=self.wav2vec2_for_ctc.parameters())
        self.scheduler_factory: partial[torch.optim.lr_scheduler] = scheduler

        self.metrics = MetricCollection(dict(char_error_rate=CharErrorRate()))
        self.unfreeze_at_step: int = unfreeze_at_step
        self.push_to_hub_after_testing: bool = push_to_hub_after_testing
        self.max_steps = nb_steps_per_epoch * max_epochs
        self.description: str = description
        
        self.num_val_runs: int = 0
        self.dataloader_names: List[str] = None

    def on_train_start(self) -> None:
        self.once = False
        if self.unfreeze_at_step > 0:
            logger.info("Entering freeze transformer layers learning strategy")
            for param in self.wav2vec2_for_ctc.wav2vec2.encoder.layers.parameters():
                param.requires_grad = False
            self.once = True

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx (int): Index of the current batch
        """
        if self.once and 0 < self.unfreeze_at_step < self.global_step:
            logger.info("Entering unfreeze transformer layers learning strategy")
            for param in self.wav2vec2_for_ctc.wav2vec2.encoder.layers.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            self.once = False

        return self.common_step(batch, batch_idx, "train", dataloader_idx=0)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        return self.common_step(batch, batch_idx, "validation", dataloader_idx)

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Lightning test step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        return self.common_step(batch, batch_idx, "test", dataloader_idx)

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """
        optimizer = self.optimizer_factory

        scheduler = {
            "scheduler": self.scheduler_factory(
            optimizer=optimizer,
            warmup_steps=int(1e-1 * self.max_steps),
            hold_steps=int(4e-1 * self.max_steps),
            decay_steps=int(5e-1 * self.max_steps),
            total_steps=self.max_steps,
        ),
            'interval': 'step',
        }

        return [optimizer], [scheduler]
    
    def _setup_metrics_and_logging(self, dataloader_type: str = "val") -> None:
        """
        Initialize metrics, validate datamodule consistency, and setup logging infrastructure.
        
        This method performs essential setup tasks at the start of training/validation/test phases:
        1. Moves all metric collections to the appropriate device (GPU/CPU)
        2. Validates that datamodule parameters match module requirements
        3. Logs model description to experiment tracker if available
        4. Extracts dataloader names for multi-dataset evaluation logging
        
        Args:
            dataloader_type (str): Type of dataloader to inspect for names.
                Options: 'val' for validation, 'test' for testing. Defaults to 'val'.
        
        Note:
            This method should be called in hooks where the trainer's datamodule is available
            (e.g., on_fit_start, on_validation_start, on_test_start).
        """
        # Move metrics to device
        for category in self.metrics:
            self.metrics[category] = self.metrics[category].to(self.device)
        
        # Validate datamodule consistency
        self.check_datamodule_parameters()
        
        # Log model description
        if self.logger and self.description:
            self.logger.experiment.add_text(
                tag="description", text_string=self.description
            )
        
        # Extract dataloader names for multi-dataset logging
        dataloader_method = f"{dataloader_type}_dataloader"
        if hasattr(self.trainer.datamodule, dataloader_method):
            dataloader = getattr(self.trainer.datamodule, dataloader_method)()
            if isinstance(dataloader, dict):
                self.dataloader_names = list(dataloader.keys())

    def on_fit_start(self) -> None:
        """
        Called at the beginning of the fit loop.

        - Checks the consistency of the DataModule's parameters
        """
        self._setup_metrics_and_logging(dataloader_type="val")
        
    def on_validation_start(self):
        """
        Called when the validation loop begins.
        """
        self._setup_metrics_and_logging(dataloader_type="val")

    def on_test_start(self):
        """
        Called when the test loop begins.
        Validates datamodule parameters and sets up dataloader names for multi-dataset testing.
        Displays frozen status of branches.
        """
        self._setup_metrics_and_logging(dataloader_type="test")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """
        Method automatically called when the train batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the training_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """

        self.common_logging("train", outputs, batch, batch_idx)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Method automatically called when the validation batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the validation_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """

        self.common_logging("validation", outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Method automatically called when the test batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the test_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """
        self.common_logging("test", outputs, batch, batch_idx, dataloader_idx)

    def on_test_end(self) -> None:
        """
        Method to be called when the test ends.
        """
        if self.push_to_hub_after_testing:
            self.wav2vec2_for_ctc.push_to_hub(
                f"Cnam-LMSSC/phonemizer_{self.trainer.datamodule.sensor}",
                commit_message=f"Upload Wav2Vec2ForCTC after {self.trainer.current_epoch} epochs",
            )
            processor = Wav2Vec2Processor(
                feature_extractor=self.trainer.datamodule.feature_extractor, tokenizer=self.trainer.datamodule.tokenizer
            )
            processor.push_to_hub(
                f"Cnam-LMSSC/phonemizer_{self.trainer.datamodule.sensor}",
                commit_message=f"Upload Wav2Vec2Processor after {self.trainer.current_epoch} epochs",
            )

    def common_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str,
        dataloader_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Common step for training, validation and test steps.

        Args:
             batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"

        Returns:
            Dict[str, torch.Tensor]: Dict with keys "loss", "logits"
        """
        dl_name_suffix = (
            f"/{self.dataloader_names[dataloader_idx]}"
            if (self.dataloader_names is not None and not stage == "train")
            else ""
        )

        # Get tensors
        speech = batch["audio"]
        target_ids = batch["phonemes_ids"]

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)
        
        # Log loss
        self.log(
            f"{stage}/loss{dl_name_suffix}",
            forward_result.loss,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return forward_result

    def common_logging(
        self, stage: str, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Common logging for training, validation and test steps.

        Args:
            stage(str): Stage of the training
            outputs(STEP_OUTPUT): Output of the {train,validation,test}_step method
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx(int): Index of the batch
            dataloader_idx(int): Index of the dataloader
        """
        dl_name_suffix = (
            f"/{self.dataloader_names[dataloader_idx]}"
            if self.dataloader_names is not None
            else ""
        )

        # Log metrics
        predicted_phonemes = self.get_phonemes_from_logits(outputs["logits"])
        target_phonemes = batch["phonemes_str"]
        metrics_to_log = self.metrics(predicted_phonemes, target_phonemes)
        metrics_to_log = {f"{stage}/{k}{dl_name_suffix}": v for k, v in metrics_to_log.items()}

        self.log_dict(
            dictionary=metrics_to_log,
            sync_dist=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        # Log text
        text_to_log = f"OUT: {predicted_phonemes[0]}" + "  \n" + f"GT:{target_phonemes[0]} "
        self.logger.experiment.add_text(
            tag=f"{stage}/predicted_vs_target__phonemes{dl_name_suffix}",
            text_string=text_to_log,
            global_step=self.trainer.global_step + batch_idx,
        )

    def get_phonemes_from_logits(self, model_logits):
        """
        Get phonemes from model logits

        Args:
            model_logits(torch.Tensor): Model logits

        Returns:
            List[str]: List of predicted phonemes
        """

        # Get predicted phonemes
        predicted_ids = torch.argmax(model_logits, dim=2)
        predicted_phonemes = [
            self.trainer.datamodule.tokenizer.decode(predicted_ids[i, :]) for i in range(predicted_ids.shape[0])
        ]

        return predicted_phonemes

    def check_datamodule_parameters(self) -> None:
        """
        List of assertions checking that the parameters of the LightningDatamodule correspond to the LightningModule.

        (Can only be called in stages where the trainer's LightningDataModule is available, e.g. in on_fit_start hook.)

        - Checks the LightningDataModule sample_rate.
        - Checks tokenizer's pad_token_id.
        - Checks the length of the tokenizer.
        """
        # Check sample rate
        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"{self.sample_rate} is specified for the LightningModule and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )

        # Check tokenizer's pad_token_id
        assert self.trainer.datamodule.tokenizer.pad_token_id == 36, "Pad token id must be 36"

        # Check length of tokenizer
        assert len(self.trainer.datamodule.tokenizer) == 39, "Vocab size must be 39"
