# main.py

import hydra 
from hydra.utils import to_absolute_path 
from omegaconf import DictConfig, OmegaConf 
import os 
import torch 
import albumentations as A 
from torch.utils.data import DataLoader 
import torch.optim as optim
from datetime import datetime

from dataset import XRayInferenceDataset, get_test_file_list 
from models import initialize_model 
from trainer import Trainer 
from inferencer import Inferencer 
from utils import set_seed, get_optimizer, get_scheduler 
from data_utils import prepare_data, create_datasets, create_dataloaders 
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss, TverskyLoss  # 다양한 Loss 추가
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Transformer 관련 임포트
from transformers import SegformerImageProcessor
from albumentations import Compose, Lambda

@hydra.main(version_base=None, config_path="config", config_name="config") 
def main(config: DictConfig): 
    # Disable structured mode to allow adding new keys 
    OmegaConf.set_struct(config, False) 

    # Add class mappings to the config 
    config.CLASS2IND = {v: i for i, v in enumerate(config.CLASSES)} 
    config.IND2CLASS = {i: v for i, v in enumerate(config.CLASSES)} 
    print("Starting main.py with Hydra configuration...") 

    # Set random seed for reproducibility 
    set_seed(config.RANDOM_SEED) 

    # Set device to GPU if available, else CPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Using device: {device}") 

    # Get the current date in 'YYYYMMDD' format
    date = datetime.now().strftime("%Y%m%d")
    
    # Retrieve optimizer and scheduler names from config
    optimizer_name = config.optimizer.name
    scheduler_name = config.scheduler.name
    
    # Retrieve loss name from config
    loss_name = config.loss.name  # <-- 손실 함수 이름 추가

    # Retrieve encoder_name and encoder_weights from config
    model_name_config = config.model.model_name
    if model_name_config == 'segformer':
        encoder_name = config.model.get('encoder_name', 'nvidia/segformer-b0')
        encoder_weights = None  # SegFormer에서는 encoder_weights를 사용하지 않음
    else:
        encoder_name = config.model.get('encoder_name', 'resnet34')
        encoder_weights = config.model.get('encoder_weights', 'imagenet')  # 기본값을 'imagenet'으로 설정

    # Extract augmentation types from config
    aug_types = [aug['type'] for aug in config.augmentations.train]
    aug_str = '_'.join(aug_types) if aug_types else 'NoAug'

    # Create experiment_name based on model information, optimizer/scheduler, loss, augmentations, and date
    model_type = 'transformer' if config.model.model_name in ['segformer'] else 'SMP'
    model_name = config.model.model_name
    # 수정된 experiment_name 형식: 모델 정보 + 증강 기법 + 기타 정보
    experiment_name = f"{model_type}_{model_name}_{encoder_name.replace('/', '_')}_{optimizer_name}_{scheduler_name}_{loss_name}_{aug_str}_{date}"

    # Set SAVED_DIR to 'checkpoints/experiment_name'
    config.SAVED_DIR = os.path.join(to_absolute_path(config.SAVED_DIR), experiment_name)

    # Create checkpoint directory
    if not os.path.exists(config.SAVED_DIR): 
        os.makedirs(config.SAVED_DIR) 
        print(f"Created directory for saving checkpoints: {config.SAVED_DIR}") 

    # Save the config.yaml file into SAVED_DIR
    config_save_path = os.path.join(config.SAVED_DIR, 'config.yaml')
    OmegaConf.save(config, config_save_path)
    print(f"Saved config file to {config_save_path}")

    # Create output directory
    output_dir = os.path.join(config.SAVED_DIR, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Generate CSV file name
    csv_filename = f"{experiment_name}.csv"
    csv_path = os.path.join(config.SAVED_DIR, csv_filename)
    print(f"CSV will be saved as: {csv_path}")

    # Determine the mode of operation: train, inference, or both 
    mode = config.get("mode", "train") 
    print(f"Mode selected: {mode}") 

    model = None  # Initialize model variable 

    if mode in ['train', 'both']: 
        print("Preparing training and validation data for Phase 1...") 
        # Phase 1: Split data into train1 (80%) and val1 (20%) 
        train_filenames_phase1, train_labelnames_phase1, val_filenames_phase1, val_labelnames_phase1 = prepare_data(config) 

        print("Creating datasets and dataloaders for Phase 1...") 
        # Get the preprocessing function based on the encoder
        if config.model.model_name == 'segformer':
            normalize = False
            feature_extractor = SegformerImageProcessor.from_pretrained(
                encoder_name,
                do_resize=True,
                size=512,  # 이미지 크기를 모델에 맞게 설정
                do_normalize=True,
                do_rescale=True
            )
            preprocessing_fn = None
        else:
            normalize = True
            if encoder_weights and encoder_weights.lower() != 'none':
                preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
            else:
                preprocessing_fn = None  # 또는 필요한 경우 별도의 전처리 함수 정의
            feature_extractor = None

        # preprocessing_fn이 None이 아니면 그대로 사용
        if preprocessing_fn is not None:
            preprocessing = preprocessing_fn
        else:
            preprocessing = None

        # Create datasets and dataloaders for Phase 1 
        train_dataset_phase1, valid_dataset_phase1 = create_datasets( 
            train_filenames_phase1, train_labelnames_phase1, val_filenames_phase1, val_labelnames_phase1, 
            config, preprocessing, feature_extractor, normalize=normalize  # 수정된 preprocessing 전달
        ) 
        train_loader_phase1, valid_loader_phase1 = create_dataloaders(train_dataset_phase1, valid_dataset_phase1, config) 

        # Display dataset sizes 
        print(f"Phase 1 - Training dataset size: {len(train_dataset_phase1)}") 
        print(f"Phase 1 - Validation dataset size: {len(valid_dataset_phase1)}") 

        print("Initializing the model...") 
        # Initialize the model with the specified architecture and parameters 
        model = initialize_model( 
            model_name=config.model.model_name,  
            num_classes=len(config.CLASSES), 
            pretrained=True, 
            aux_loss=config.model.aux_loss,  
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            config=config  # config 전달
        )
        model = model.to(device) 
        print(f"Model '{config.model.model_name}' initialized with encoder '{encoder_name}' and moved to {device}.")  

        print("Setting up training components for Phase 1...") 
        # Define loss function 
        if config.loss.name == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss() 
        elif config.loss.name == 'dice':
            criterion = DiceLoss(mode='multilabel')  # DiceLoss 적용
        elif config.loss.name == 'jaccard':
            criterion = JaccardLoss(mode='multilabel')  # Jaccard Loss 추가
        elif config.loss.name == 'focal':
            criterion = FocalLoss(mode='multilabel')  # Focal Loss 추가
        elif config.loss.name == 'tversky':
            criterion = TverskyLoss(mode='multilabel')  # Tversky Loss 추가
        else:
            raise ValueError(f"Unknown loss function specified: {config.loss.name}")

        # Initialize optimizer based on config
        optimizer = get_optimizer(
            name=config.optimizer.name,
            config=config,
            parameters=model.parameters()
        )
        print(f"Optimizer '{optimizer_name}' initialized.")

        # Initialize scheduler based on config
        scheduler = None
        if config.scheduler.name.lower() != 'none':
            scheduler = get_scheduler(
                name=config.scheduler.name,
                config=config,
                optimizer=optimizer
            )
            # Print scheduler parameters
            if isinstance(scheduler, optim.lr_scheduler.StepLR):
                print(f"Scheduler '{scheduler_name}' initialized with step_size={scheduler.step_size}, gamma={scheduler.gamma}")
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                print(f"Scheduler '{scheduler_name}' initialized with mode={scheduler.mode}, factor={scheduler.factor}, patience={scheduler.patience}")
            elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                print(f"Scheduler '{scheduler_name}' initialized with T_max={scheduler.T_max}, eta_min={scheduler.eta_min}")
            else:
                print(f"Scheduler '{scheduler_name}' initialized.")

        # Initialize the Trainer with the defined components 
        trainer_phase1 = Trainer( 
            model=model, 
            model_name=config.model.model_name,
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=device, 
            config=config, 
            threshold=config.get("threshold", 0.5),
            phase="phase1"  # Phase 1 identifier for checkpoint naming
        ) 

        print("Starting Phase 1 training process...") 
        # Start the Phase 1 training process 
        trainer_phase1.train(train_loader_phase1, valid_loader_phase1, num_epochs=config.NUM_EPOCHS_PHASE1) 
        print("Phase 1 training completed successfully.") 

        # Phase 2: Swap train and val 
        print("Preparing training and validation data for Phase 2 by swapping Phase 1 splits...") 
        train_filenames_phase2 = val_filenames_phase1  # Phase 1's val is Phase 2's train 
        train_labelnames_phase2 = val_labelnames_phase1 
        val_filenames_phase2 = train_filenames_phase1  # Phase 1's train is Phase 2's val 
        val_labelnames_phase2 = train_labelnames_phase1 

        print("Creating datasets and dataloaders for Phase 2...") 
        # Create datasets and dataloaders for Phase 2 
        train_dataset_phase2, valid_dataset_phase2 = create_datasets( 
            train_filenames_phase2, train_labelnames_phase2, val_filenames_phase2, val_labelnames_phase2, 
            config, preprocessing, feature_extractor, normalize=normalize  # 수정된 preprocessing 전달
        ) 
        train_loader_phase2, valid_loader_phase2 = create_dataloaders(train_dataset_phase2, valid_dataset_phase2, config) 

        # Display dataset sizes 
        print(f"Phase 2 - Training dataset size: {len(train_dataset_phase2)}") 
        print(f"Phase 2 - Validation dataset size: {len(valid_dataset_phase2)}") 

        # Calculate Phase 2 epochs 
        num_epochs_phase2 = int(config.NUM_EPOCHS_PHASE1 * config.PHASE2_RATIO) 
        print(f"Phase 2 will run for {num_epochs_phase2} epochs.") 

        print("Setting up training components for Phase 2...") 
        # Optionally, adjust optimizer or other components for Phase 2 
        # For simplicity, we'll continue with the same optimizer and model 

        # Initialize the Trainer for Phase 2 
        trainer_phase2 = Trainer( 
            model=model, 
            model_name=config.model.model_name,
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            device=device, 
            config=config, 
            threshold=config.get("threshold", 0.5),
            phase="phase2"  # Phase 2 identifier for checkpoint naming
        ) 

        print("Starting Phase 2 training process...") 
        # Start the Phase 2 training process 
        trainer_phase2.train(train_loader_phase2, valid_loader_phase2, num_epochs=num_epochs_phase2) 
        print("Phase 2 training completed successfully.") 

    if mode in ['inference', 'both']: 
        print("Starting inference process...") 
        # Load the model if it hasn't been initialized during training 
        if model is None: 
            model_files = [f for f in os.listdir(config.SAVED_DIR) if f.endswith('.pt')] 
            if not model_files:
                raise FileNotFoundError( 
                    "No saved model files found. Please specify 'model_path' in the config." 
                ) 
            # 수정된 부분: 모델 파일명에서 dice score를 추출하여 가장 높은 것을 선택
            model_files_with_scores = []
            for f in model_files:
                try:
                    parts = f.replace('.pt', '').split('_')
                    dice_index = parts.index('dice')
                    val_dice_str = parts[dice_index + 1]
                    val_dice = float(val_dice_str)
                    model_files_with_scores.append((val_dice, f))
                except (ValueError, IndexError) as e:
                    print(f"Filename {f} does not match expected format. Skipping.")
                    continue
            if not model_files_with_scores:
                raise FileNotFoundError("No valid model files with dice scores found.")
            # dice score를 기준으로 내림차순 정렬하여 가장 높은 모델 선택
            model_files_with_scores.sort(key=lambda x: x[0], reverse=True)
            best_val_dice, best_model_file = model_files_with_scores[0]
            model_path = os.path.join(config.SAVED_DIR, best_model_file)
            print(f"Loading model with highest validation dice score: {model_path}, dice score: {best_val_dice}")

            # Initialize the model architecture without pre-trained weights 
            model = initialize_model( 
                model_name=config.model.model_name,  
                num_classes=len(config.CLASSES), 
                pretrained=False, 
                aux_loss=config.model.aux_loss,  
                encoder_name=encoder_name,
                encoder_weights=None,  # Inference 시에는 가중치를 로드하므로 None으로 설정
                config=config  # config 전달
            ) 
            # Load the saved model weights 
            model.load_state_dict(torch.load(model_path, map_location=device)) 
            model = model.to(device) 
            print(f"Model '{config.model.model_name}' loaded from {model_path} and moved to {device}.")  

        # Get the preprocessing function based on the encoder
        if config.model.model_name == 'segformer':
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
            if encoder_weights and encoder_weights.lower() != 'none':
                preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
            else:
                preprocessing_fn = None  # 또는 필요한 경우 별도의 전처리 함수 정의
            feature_extractor = None

        # preprocessing_fn이 None이 아니면 그대로 사용
        if preprocessing_fn is not None:
            preprocessing = preprocessing_fn
        else:
            preprocessing = None

        print("Preparing test data for inference...") 
        # Get the list of test image files 
        test_pngs = get_test_file_list(config.TEST_IMAGE_ROOT) 

        if len(test_pngs) == 0: 
            print("No test image files found. Please check the 'TEST_IMAGE_ROOT' path.") 
            return 

        print("Creating test dataset and dataloaders...") 
        # Define transformations for the test dataset 
        test_transforms = A.Compose([
            A.Resize(height=1024, width=1024)
            # 필요한 경우 추가 변환
        ])

        # Create test dataset with preprocessing_fn
        test_dataset = XRayInferenceDataset( 
            filenames=test_pngs, 
            image_root=config.TEST_IMAGE_ROOT, 
            transforms=test_transforms,
            preprocessing=preprocessing,
            feature_extractor=feature_extractor,
            normalize=normalize  # normalize 인자 추가
        ) 
        test_loader = DataLoader( 
            dataset=test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=config.get("num_workers", 8) 
        ) 
        print(f"Number of test samples: {len(test_dataset)}") 

        print("Initializing inferencer...") 
        # Initialize the Inferencer with the model and configuration 
        inferencer = Inferencer( 
            model=model, 
            model_name=config.model.model_name,
            device=device, 
            config=config, 
            threshold=config.get("threshold", 0.5), 
            output_size=tuple(config.get("output_size", [2048, 2048])) 
        ) 

        print("Running inference on test data...") 
        # Perform inference 
        rles, filename_and_class = inferencer.inference(test_loader) 
        print("Inference completed successfully.") 

        print("Saving inference results...") 
        # Define the output path for the results
        # Set the CSV file name to the same experiment_name
        output_filename = csv_filename  
        output_path = os.path.join(output_dir, output_filename)
        inferencer.save_results(rles, filename_and_class, output_path) 
        print(f"Inference results saved to {output_path}") 

if __name__ == "__main__": 
    main()
