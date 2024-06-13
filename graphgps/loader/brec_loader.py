from brec.dataset import BRECDataset

def load_brec_dataset(
    format : str,
    name : str,
    dataset_dir : str
):
    if format == "Custom":
        if name == "BREC":
            return BRECDataset(root=dataset_dir)
                
