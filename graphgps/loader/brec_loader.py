def load_brec_dataset(
    format : str,
    name : str,
    dataset_dir : str
):
    if format == "Custom":
        if name == "BREC":
            from brec.dataset import BRECDataset
            return BRECDataset(root=dataset_dir)
                
