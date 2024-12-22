# main.py
import hydra 
from hydra.utils import to_absolute_path 
from omegaconf import DictConfig, OmegaConf 
import os 
import torch
import albumentations as A
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import XRayInferenceDataset, get_test_file_list
from models import initialize_model
from trainer import Trainer
from inferencer import Inferencer
from utils import set_seed, get_optimizer, get_scheduler
from data_utils import prepare_data, create_datasets, create_dataloaders
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss, TverskyLoss
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from transformers import SegformerImageProcessor

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """
    Hydra 설정 기반으로 전체 파이프라인(학습/검증/추론)을 실행하는 메인 함수.
    """
    OmegaConf.set_struct(config, False)

    config.CLASS2IND = {v: i for i, v in enumerate(config.CLASSES)}
    config.IND2CLASS = {i: v for i, v in enumerate(config.CLASSES)}
    print("Starting main.py with Hydra configuration...")

    set_seed(config.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    date = datetime.now().strftime("%Y%m%d")
    model_name_config = config.model.model_name
    optimizer_name = config.optimizer.name
    scheduler_name = config.scheduler.name
    loss_name = config.loss.name
    if model_name_config == 'segformer':
        encoder_name = config.model.get('encoder_name', 'nvidia/segformer-b0')
        encoder_weights = None
    else:
        encoder_name = config.model.get('encoder_name', 'resnet34')
        encoder_weights = config.model.get('encoder_weights', 'imagenet')

    aug_types = [aug['type'] for aug in config.augmentations.train]
    aug_str = '_'.join(aug_types) if aug_types else 'NoAug'
    model_type = 'transformer' if model_name_config in ['segformer'] else 'SMP'
    experiment_name = f"{model_type}_{model_name_config}_{encoder_name.replace('/', '_')}_{optimizer_name}_{scheduler_name}_{loss_name}_{aug_str}_{date}"

    config.SAVED_DIR = os.path.join(to_absolute_path(config.SAVED_DIR), experiment_name)
    os.makedirs(config.SAVED_DIR, exist_ok=True)
    print(f"Checkpoints directory: {config.SAVED_DIR}")

    config_save_path = os.path.join(config.SAVED_DIR, 'config.yaml')
    OmegaConf.save(config, config_save_path)
    print(f"Saved config file to {config_save_path}")

    output_dir = os.path.join(config.SAVED_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    csv_filename = f"{experiment_name}.csv"
    csv_path = os.path.join(config.SAVED_DIR, csv_filename)
    print(f"CSV will be saved as: {csv_path}")

    mode = config.get("mode", "train")
    print(f"Mode selected: {mode}")

    model = None

    # Training Phase
    if mode in ['train', 'both']:
        print("Preparing training and validation data for Phase 1...")
        train_filenames_phase1, train_labelnames_phase1, val_filenames_phase1, val_labelnames_phase1 = prepare_data(config)

        print("Creating datasets and dataloaders for Phase 1...")
        if model_name_config == 'segformer':
            normalize = False
            feature_extractor = SegformerImageProcessor.from_pretrained(
                encoder_name,
                do_resize=True,
                size=512,
                do_normalize=True,
                do_rescale=True
            )
            preprocessing_fn = None
        else:
            normalize = True
            preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights) if encoder_weights and encoder_weights.lower() != 'none' else None
            feature_extractor = None

        preprocessing = preprocessing_fn if preprocessing_fn is not None else None

        train_dataset_phase1, valid_dataset_phase1 = create_datasets(
            train_filenames_phase1, train_labelnames_phase1,
            val_filenames_phase1, val_labelnames_phase1,
            config, preprocessing, feature_extractor, normalize=normalize
        )
        train_loader_phase1, valid_loader_phase1 = create_dataloaders(train_dataset_phase1, valid_dataset_phase1, config)

        print(f"Phase 1 - Train size: {len(train_dataset_phase1)}, Val size: {len(valid_dataset_phase1)}")
        print("Initializing the model...")
        model = initialize_model(model_name_config, len(config.CLASSES), True, config.model.aux_loss, encoder_name, encoder_weights, config)
        model = model.to(device)
        print(f"Model '{model_name_config}' initialized.")

        print("Setting up training components for Phase 1...")
        if loss_name == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif loss_name == 'dice':
            criterion = DiceLoss(mode='multilabel')
        elif loss_name == 'jaccard':
            criterion = JaccardLoss(mode='multilabel')
        elif loss_name == 'focal':
            criterion = FocalLoss(mode='multilabel')
        elif loss_name == 'tversky':
            criterion = TverskyLoss(mode='multilabel')
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        optimizer = get_optimizer(optimizer_name, config, model.parameters())
        scheduler = None
        if scheduler_name.lower() != 'none':
            scheduler = get_scheduler(scheduler_name, config, optimizer)

        trainer_phase1 = Trainer(model, model_name_config, criterion, optimizer, scheduler, device, config, threshold=config.get("threshold", 0.5), phase="phase1")

        print("Starting Phase 1 training...")
        trainer_phase1.train(train_loader_phase1, valid_loader_phase1, num_epochs=config.NUM_EPOCHS_PHASE1)
        print("Phase 1 training completed.")

        print("Preparing data for Phase 2 by swapping Phase 1 splits...")
        train_filenames_phase2 = val_filenames_phase1
        train_labelnames_phase2 = val_labelnames_phase1
        val_filenames_phase2 = train_filenames_phase1
        val_labelnames_phase2 = train_labelnames_phase1

        train_dataset_phase2, valid_dataset_phase2 = create_datasets(
            train_filenames_phase2, train_labelnames_phase2,
            val_filenames_phase2, val_labelnames_phase2,
            config, preprocessing, feature_extractor, normalize=normalize
        )
        train_loader_phase2, valid_loader_phase2 = create_dataloaders(train_dataset_phase2, valid_dataset_phase2, config)

        print(f"Phase 2 - Train size: {len(train_dataset_phase2)}, Val size: {len(valid_dataset_phase2)}")
        num_epochs_phase2 = int(config.NUM_EPOCHS_PHASE1 * config.PHASE2_RATIO)
        print(f"Phase 2 will run for {num_epochs_phase2} epochs.")

        trainer_phase2 = Trainer(model, model_name_config, criterion, optimizer, scheduler, device, config, threshold=config.get("threshold", 0.5), phase="phase2")
        print("Starting Phase 2 training...")
        trainer_phase2.train(train_loader_phase2, valid_loader_phase2, num_epochs=num_epochs_phase2)
        print("Phase 2 training completed.")

    # Inference Phase
    if mode in ['inference', 'both']:
        print("Starting inference process...")
        if model is None:
            model_files = [f for f in os.listdir(config.SAVED_DIR) if f.endswith('.pt')]
            if not model_files:
                raise FileNotFoundError("No saved model files found.")

            model_files_with_scores = []
            for f in model_files:
                parts = f.replace('.pt', '').split('_')
                if 'dice' in parts:
                    dice_index = parts.index('dice')
                    val_dice_str = parts[dice_index + 1]
                    try:
                        val_dice = float(val_dice_str)
                        model_files_with_scores.append((val_dice, f))
                    except:
                        pass

            if not model_files_with_scores:
                raise FileNotFoundError("No valid model files with dice scores found.")

            model_files_with_scores.sort(key=lambda x: x[0], reverse=True)
            best_val_dice, best_model_file = model_files_with_scores[0]
            model_path = os.path.join(config.SAVED_DIR, best_model_file)
            print(f"Loading model {model_path} with dice {best_val_dice}")
            model = initialize_model(model_name_config, len(config.CLASSES), False, config.model.aux_loss, encoder_name, None, config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

        if model_name_config == 'segformer':
            normalize = False
            feature_extractor = SegformerImageProcessor.from_pretrained(
                encoder_name,
                do_resize=True,
                size=512,
                do_normalize=True,
                do_rescale=True
            )
            preprocessing_fn = None
        else:
            normalize = True
            preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights) if encoder_weights and encoder_weights.lower() != 'none' else None
            feature_extractor = None

        preprocessing = preprocessing_fn if preprocessing_fn else None
        print("Preparing test data for inference...")
        test_pngs = get_test_file_list(config.TEST_IMAGE_ROOT)
        if len(test_pngs) == 0:
            print("No test images found.")
            return

        test_transforms = A.Compose([A.Resize(height=512, width=512)])
        test_dataset = XRayInferenceDataset(
            filenames=test_pngs,
            image_root=config.TEST_IMAGE_ROOT,
            transforms=test_transforms,
            preprocessing=preprocessing,
            feature_extractor=feature_extractor,
            normalize=normalize
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=config.get("num_workers", 8)
        )
        print(f"Number of test samples: {len(test_dataset)}")

        inferencer = Inferencer(model, model_name_config, device, config, threshold=config.get("threshold", 0.5), output_size=tuple(config.get("output_size", [2048, 2048])))
        print("Running inference...")
        rles, filename_and_class = inferencer.inference(test_loader)
        print("Inference completed.")

        output_filename = csv_filename
        output_path = os.path.join(output_dir, output_filename)
        inferencer.save_results(rles, filename_and_class, output_path)
        print(f"Inference results saved to {output_path}")

if __name__ == "__main__":
    main()
