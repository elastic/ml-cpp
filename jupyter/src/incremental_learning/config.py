from pathlib2 import Path

root_dir = Path(__file__).parent.parent.parent.absolute()
data_dir = root_dir / 'data'
datasets_dir = data_dir / 'datasets'
configs_dir = data_dir / 'configs'