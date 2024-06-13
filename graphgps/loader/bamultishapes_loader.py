from torch_geometric.graphgym.register import register_loader
from torch_geometric.datasets import BAMultiShapesDataset

def load_bamultishapes_dataset(
    format : str,
    name : str,
    dataset_dir : str
):
    if format == "Custom":
        if name == "BAMultiShapes":
            dataset_dir = f"{dataset_dir}/{name}"
            return BAMultiShapesDataset(root=dataset_dir)
    

if __name__=="__main__":
    dataset = load_bamultishapes_dataset(
        format="PyG",
        name="BAMultiShapes",
        dataset_dir="\tmp"
    )
    print(len(dataset))
    print(dataset[0])