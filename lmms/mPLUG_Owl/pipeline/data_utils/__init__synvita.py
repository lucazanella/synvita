import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
# from .processors.builder import build_processors
from lmms.mPLUG_Owl.mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

def train_valid_test_datasets_provider(data_dir, features_dir, data_path, config, tokenizer, seq_length=1024, loss_objective = 'sequential', use_extracted_features=False):
    """Build train and valid datasets."""
    print('> building train and validation datasets for mPLUG-Owl ...')
    train_ds, valid_ds = build_train_valid_test_datasets(
        data_dir=data_dir,
        features_dir=features_dir,
        input_file=data_path,  
        tokenizer=tokenizer,
        max_length=seq_length, 
        config=config, loss_objective = loss_objective, use_extracted_features=use_extracted_features)
    print("> finished creating mPLUG-Owl datasets ...")

    return train_ds, valid_ds


def build_train_valid_test_datasets(data_dir, features_dir, input_file, tokenizer, max_length=80, config=None, loss_objective = 'sequential', use_extracted_features=False, minibatch_type="default", remove_not_mask=False, source=None, use_best_synthetic=False, use_semantic_regularization=True, use_lm_semantic_regularization=False):
    
    # train_processors = build_processors(config['train_processors'])
    # valid_processors = build_processors(config['valid_processors'])

    image_processor = MplugOwlImageProcessor.from_pretrained(config['pretrained_ckpt'])
    processor = MplugOwlProcessor(image_processor, tokenizer)
    
    assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
    from .xgpt3_dataset_synvita import MultiModalDataset
    train_ds = MultiModalDataset(data_dir, features_dir, input_file[0], tokenizer, processor, max_length, loss_objective = loss_objective, use_extracted_features=use_extracted_features)
    from .xgpt3_dataset import MultiModalDataset
    valid_ds = MultiModalDataset(data_dir, features_dir, input_file[1], tokenizer, processor, max_length, loss_objective = loss_objective, use_extracted_features=use_extracted_features)    
    return (train_ds, valid_ds)
